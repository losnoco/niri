use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::element::{Element, Id, Kind, RenderElement, UnderlyingStorage};
use smithay::backend::renderer::gles::{GlesError, GlesFrame, GlesRenderer, GlesTexture};
use smithay::backend::renderer::utils::{CommitCounter, DamageSet, OpaqueRegions};
use smithay::backend::renderer::{
    ContextId, ErasedContextId, Frame as _, ImportMem, Renderer, Texture,
};
use smithay::utils::user_data::UserDataMap;
use smithay::utils::{Buffer, Logical, Physical, Point, Rectangle, Scale, Size, Transform};

use super::memory::MemoryBuffer;
use super::renderer::AsGlesFrame as _;
use crate::backend::tty::{TtyFrame, TtyRenderer, TtyRendererError};
use crate::backend::tty_renderer::TtyOffscreen;

/// Smithay's texture buffer, but with fractional scale.
#[derive(Debug, Clone)]
pub struct TextureBuffer<T: Texture> {
    id: Id,
    commit_counter: CommitCounter,
    renderer_context_id: ErasedContextId,
    texture: T,
    scale: Scale<f64>,
    transform: Transform,
    opaque_regions: Vec<Rectangle<i32, Buffer>>,
}

/// Render element for a [`TextureBuffer`].
#[derive(Debug, Clone)]
pub struct TextureRenderElement<T: Texture> {
    buffer: TextureBuffer<T>,
    location: Point<f64, Logical>,
    alpha: f32,
    src: Option<Rectangle<f64, Logical>>,
    size: Option<Size<f64, Logical>>,
    kind: Kind,
}

impl<T: Texture + 'static> TextureBuffer<T> {
    pub fn from_texture<R: Renderer>(
        renderer: &R,
        texture: T,
        scale: impl Into<Scale<f64>>,
        transform: Transform,
        opaque_regions: Vec<Rectangle<i32, Buffer>>,
    ) -> Self
    where
        R::TextureId: 'static,
    {
        TextureBuffer {
            id: Id::new(),
            commit_counter: CommitCounter::default(),
            renderer_context_id: ContextId::erased(&renderer.context_id()),
            texture,
            scale: scale.into(),
            transform,
            opaque_regions,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_memory<R: Renderer<TextureId = T> + ImportMem>(
        renderer: &mut R,
        data: &[u8],
        format: Fourcc,
        size: impl Into<Size<i32, Buffer>>,
        flipped: bool,
        scale: impl Into<Scale<f64>>,
        transform: Transform,
        opaque_regions: Vec<Rectangle<i32, Buffer>>,
    ) -> Result<Self, R::Error> {
        let texture = renderer.import_memory(data, format, size.into(), flipped)?;
        Ok(TextureBuffer::from_texture(
            renderer,
            texture,
            scale,
            transform,
            opaque_regions,
        ))
    }

    pub fn from_memory_buffer<R: Renderer<TextureId = T> + ImportMem>(
        renderer: &mut R,
        buffer: &MemoryBuffer,
    ) -> Result<Self, R::Error> {
        Self::from_memory(
            renderer,
            buffer.data(),
            buffer.format(),
            buffer.size(),
            false,
            buffer.scale(),
            buffer.transform(),
            Vec::new(),
        )
    }

    pub fn texture(&self) -> &T {
        &self.texture
    }

    /// Converts the texture type, keeping all buffer metadata.
    pub fn map_texture<U: Texture>(self, f: impl FnOnce(T) -> U) -> TextureBuffer<U> {
        TextureBuffer {
            id: self.id,
            commit_counter: self.commit_counter,
            renderer_context_id: self.renderer_context_id,
            texture: f(self.texture),
            scale: self.scale,
            transform: self.transform,
            opaque_regions: self.opaque_regions,
        }
    }

    pub fn texture_scale(&self) -> Scale<f64> {
        self.scale
    }

    pub fn set_texture_scale(&mut self, scale: impl Into<Scale<f64>>) {
        self.scale = scale.into();
    }

    pub fn texture_transform(&self) -> Transform {
        self.transform
    }

    pub fn set_texture_transform(&mut self, transform: Transform) {
        self.transform = transform;
    }
}

impl<T: Texture> TextureBuffer<T> {
    pub fn logical_size(&self) -> Size<f64, Logical> {
        self.texture
            .size()
            .to_f64()
            .to_logical(self.scale, self.transform)
    }
}

impl TextureBuffer<GlesTexture> {
    pub fn is_texture_reference_unique(&mut self) -> bool {
        self.texture.is_unique_reference()
    }
}

impl<T: Texture> TextureRenderElement<T> {
    pub fn from_texture_buffer(
        buffer: TextureBuffer<T>,
        location: impl Into<Point<f64, Logical>>,
        alpha: f32,
        src: Option<Rectangle<f64, Logical>>,
        size: Option<Size<f64, Logical>>,
        kind: Kind,
    ) -> Self {
        TextureRenderElement {
            buffer,
            location: location.into(),
            alpha,
            src,
            size,
            kind,
        }
    }

    pub fn buffer(&self) -> &TextureBuffer<T> {
        &self.buffer
    }
}

impl<T: Texture> TextureRenderElement<T> {
    pub fn logical_size(&self) -> Size<f64, Logical> {
        self.size
            .or_else(|| self.src.map(|src| src.size))
            .unwrap_or_else(|| self.buffer.logical_size())
    }

    pub fn logical_src(&self) -> Rectangle<f64, Logical> {
        self.src
            .unwrap_or_else(|| Rectangle::from_size(self.logical_size()))
    }
}

impl<T: Texture> Element for TextureRenderElement<T> {
    fn id(&self) -> &Id {
        &self.buffer.id
    }

    fn current_commit(&self) -> CommitCounter {
        self.buffer.commit_counter
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        let logical_geo = Rectangle::new(self.location, self.logical_size());
        logical_geo.to_physical_precise_round(scale)
    }

    fn transform(&self) -> Transform {
        self.buffer.transform
    }

    fn src(&self) -> Rectangle<f64, Buffer> {
        self.src
            .map(|src| {
                src.to_buffer(
                    self.buffer.scale,
                    self.buffer.transform,
                    &self.buffer.logical_size(),
                )
            })
            .unwrap_or_else(|| Rectangle::from_size(self.buffer.texture.size()).to_f64())
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        let texture_size = self.buffer.texture.size().to_f64();
        let src = self.src();

        self.buffer
            .opaque_regions
            .iter()
            .filter_map(|region| {
                let mut region = region.to_f64().intersection(src)?;

                region.loc -= src.loc;
                region = region.upscale(texture_size / src.size);

                let logical =
                    region.to_logical(self.buffer.scale, self.buffer.transform, &src.size);
                Some(logical.to_physical_precise_down(scale))
            })
            .collect()
    }

    fn alpha(&self) -> f32 {
        self.alpha
    }

    fn kind(&self) -> Kind {
        self.kind
    }
}

impl<R, T> RenderElement<R> for TextureRenderElement<T>
where
    R: Renderer<TextureId = T>,
    T: Texture + 'static,
{
    fn draw(
        &self,
        frame: &mut R::Frame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dest: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        _cache: Option<&UserDataMap>,
    ) -> Result<(), R::Error> {
        if ContextId::erased(&frame.context_id()) != self.buffer.renderer_context_id {
            warn!("trying to render texture from different renderer");
            return Ok(());
        }

        frame.render_texture_from_to(
            &self.buffer.texture,
            src,
            dest,
            damage,
            opaque_regions,
            self.buffer.transform,
            self.alpha,
        )
    }

    fn underlying_storage(&self, _renderer: &mut R) -> Option<UnderlyingStorage<'_>> {
        None
    }
}

/// Render element for a [`TextureBuffer`] of the universal [`TtyOffscreen`] texture enum.
///
/// Draws into both the winit `GlesRenderer` and either TTY renderer variant, matching the
/// texture arm to the frame. Mismatched combinations (e.g. a texture captured on a different
/// backend) skip drawing with a warning.
#[derive(Debug, Clone)]
pub struct UniversalTextureRenderElement(pub TextureRenderElement<TtyOffscreen>);

impl Element for UniversalTextureRenderElement {
    fn id(&self) -> &Id {
        self.0.id()
    }

    fn current_commit(&self) -> CommitCounter {
        self.0.current_commit()
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        self.0.geometry(scale)
    }

    fn transform(&self) -> Transform {
        self.0.transform()
    }

    fn src(&self) -> Rectangle<f64, Buffer> {
        self.0.src()
    }

    fn damage_since(
        &self,
        scale: Scale<f64>,
        commit: Option<CommitCounter>,
    ) -> DamageSet<i32, Physical> {
        self.0.damage_since(scale, commit)
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        self.0.opaque_regions(scale)
    }

    fn alpha(&self) -> f32 {
        self.0.alpha()
    }

    fn kind(&self) -> Kind {
        self.0.kind()
    }
}

impl RenderElement<GlesRenderer> for UniversalTextureRenderElement {
    fn draw(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dest: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        _cache: Option<&UserDataMap>,
    ) -> Result<(), GlesError> {
        let buffer = self.0.buffer();
        let TtyOffscreen::Gles(texture) = buffer.texture() else {
            warn!("trying to render non-GLES universal texture with the GLES renderer");
            return Ok(());
        };

        frame.render_texture_from_to(
            texture,
            src,
            dest,
            damage,
            opaque_regions,
            buffer.texture_transform(),
            self.0.alpha(),
            None,
            &[],
        )
    }

    fn underlying_storage(&self, _renderer: &mut GlesRenderer) -> Option<UnderlyingStorage<'_>> {
        None
    }
}

impl<'render> RenderElement<TtyRenderer<'render>> for UniversalTextureRenderElement {
    fn draw(
        &self,
        frame: &mut TtyFrame<'render, '_, '_>,
        src: Rectangle<f64, Buffer>,
        dest: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        _cache: Option<&UserDataMap>,
    ) -> Result<(), TtyRendererError<'render>> {
        let buffer = self.0.buffer();
        let transform = buffer.texture_transform();
        let alpha = self.0.alpha();

        match (&mut *frame, buffer.texture()) {
            (frame, TtyOffscreen::Multi(texture)) => frame.render_texture_from_to(
                texture,
                src,
                dest,
                damage,
                opaque_regions,
                transform,
                alpha,
            ),
            (TtyFrame::Gles(_), TtyOffscreen::Gles(texture)) => {
                let Some(gles_frame) = frame.as_gles_frame() else {
                    return Ok(());
                };
                gles_frame
                    .render_texture_from_to(
                        texture,
                        src,
                        dest,
                        damage,
                        opaque_regions,
                        transform,
                        alpha,
                        None,
                        &[],
                    )
                    .map_err(Into::into)
            }
            (TtyFrame::Vulkan(multi), TtyOffscreen::Vulkan(texture)) => {
                let vk_frame: &mut smithay::backend::renderer::vulkan::VulkanFrame<'_, '_> =
                    multi.as_mut();
                vk_frame
                    .render_texture_from_to(
                        texture,
                        src,
                        dest,
                        damage,
                        opaque_regions,
                        transform,
                        alpha,
                    )
                    .map_err(|err| {
                        TtyRendererError::Vulkan(
                            smithay::backend::renderer::multigpu::Error::Render(err),
                        )
                    })
            }
            _ => {
                warn!("universal texture arm does not match the TTY renderer variant");
                Ok(())
            }
        }
    }

    fn underlying_storage(
        &self,
        _renderer: &mut TtyRenderer<'render>,
    ) -> Option<UnderlyingStorage<'_>> {
        None
    }
}
