//! Renderer of the TTY backend: either GLES or Vulkan.
//!
//! Wraps the two possible [`MultiRenderer`] instantiations in enums implementing the
//! rendering traits by delegation. Both variants share [`MultiTexture`] as their texture
//! type, so only the frame, framebuffer, error and texture-mapping types need wrapping.
//!
//! The GLES-specific parts of niri (custom shaders, offscreen effects) reach the raw
//! [`GlesRenderer`] through [`AsGlesRenderer`](super::super::render_helpers::renderer::AsGlesRenderer),
//! which returns `None` on the Vulkan variant; the effects degrade gracefully in that case.

use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::format::FormatSet;
use smithay::backend::allocator::Fourcc;
use smithay::backend::egl::display::EGLBufferReader;
use smithay::backend::egl::Error as EglError;
use smithay::backend::renderer::gles::{GlesRenderer, GlesTexture};
use smithay::backend::renderer::multigpu::gbm::GbmGlesBackend;
use smithay::backend::renderer::multigpu::vulkan::VulkanBackend;
use smithay::backend::renderer::multigpu::{
    Error as MultiError, MultiFrame, MultiFramebuffer, MultiRenderer, MultiTexture,
    MultiTextureMapping,
};
use smithay::backend::renderer::sync::SyncPoint;
use smithay::backend::renderer::{
    Bind, Color32F, ContextId, DebugFlags, ExportMem, Frame, ImportDma, ImportDmaWl, ImportEgl,
    ImportMem, ImportMemWl, Offscreen, Renderer, RendererSuper, Texture, TextureFilter,
    TextureMapping,
};
use smithay::reexports::wayland_server::protocol::wl_buffer::WlBuffer;
use smithay::utils::{
    Buffer as BufferCoord, DeviceFd, Physical, Point, Rectangle, Scale, Size, Transform,
};

pub type GlesApi = GbmGlesBackend<GlesRenderer, DeviceFd>;
pub type VulkanApi = VulkanBackend<DeviceFd>;

pub type GlesMultiRenderer<'render> = MultiRenderer<'render, 'render, GlesApi, GlesApi>;
pub type VulkanMultiRenderer<'render> = MultiRenderer<'render, 'render, VulkanApi, VulkanApi>;
pub type GlesMultiFrame<'render, 'frame, 'buffer> =
    MultiFrame<'render, 'render, 'frame, 'buffer, GlesApi, GlesApi>;
pub type VulkanMultiFrame<'render, 'frame, 'buffer> =
    MultiFrame<'render, 'render, 'frame, 'buffer, VulkanApi, VulkanApi>;

/// Renderer of the TTY backend.
pub enum TtyRenderer<'render> {
    Gles(GlesMultiRenderer<'render>),
    Vulkan(VulkanMultiRenderer<'render>),
}

/// Frame of the TTY backend renderer.
#[allow(clippy::large_enum_variant)]
pub enum TtyFrame<'render, 'frame, 'buffer> {
    Gles(GlesMultiFrame<'render, 'frame, 'buffer>),
    Vulkan(VulkanMultiFrame<'render, 'frame, 'buffer>),
}

/// Framebuffer of the TTY backend renderer.
#[derive(Debug)]
pub enum TtyFramebuffer<'buffer> {
    Gles(MultiFramebuffer<'buffer, GlesApi>),
    Vulkan(MultiFramebuffer<'buffer, VulkanApi>),
}

/// Texture mapping of the TTY backend renderer.
#[derive(Debug)]
pub enum TtyTextureMapping {
    Gles(MultiTextureMapping<GlesApi, GlesApi>),
    Vulkan(MultiTextureMapping<VulkanApi, VulkanApi>),
}

/// Error of the TTY backend renderer.
#[derive(Debug)]
pub enum TtyRendererError {
    Gles(MultiError<GlesApi, GlesApi>),
    Vulkan(MultiError<VulkanApi, VulkanApi>),
    VulkanUnsupported,
}

impl std::fmt::Display for TtyRendererError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TtyRendererError::Gles(err) => err.fmt(f),
            TtyRendererError::Vulkan(err) => err.fmt(f),
            TtyRendererError::VulkanUnsupported => {
                write!(f, "operation not supported by the vulkan renderer")
            }
        }
    }
}

impl std::error::Error for TtyRendererError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TtyRendererError::Gles(err) => Some(err),
            TtyRendererError::Vulkan(err) => Some(err),
            TtyRendererError::VulkanUnsupported => None,
        }
    }
}

impl From<MultiError<GlesApi, GlesApi>> for TtyRendererError {
    fn from(err: MultiError<GlesApi, GlesApi>) -> Self {
        TtyRendererError::Gles(err)
    }
}

impl From<MultiError<VulkanApi, VulkanApi>> for TtyRendererError {
    fn from(err: MultiError<VulkanApi, VulkanApi>) -> Self {
        TtyRendererError::Vulkan(err)
    }
}

impl From<smithay::backend::renderer::gles::GlesError> for TtyRendererError {
    fn from(err: smithay::backend::renderer::gles::GlesError) -> Self {
        TtyRendererError::Gles(err.into())
    }
}

impl std::fmt::Debug for TtyRenderer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TtyRenderer::Gles(renderer) => {
                f.debug_tuple("TtyRenderer::Gles").field(renderer).finish()
            }
            TtyRenderer::Vulkan(renderer) => f
                .debug_tuple("TtyRenderer::Vulkan")
                .field(renderer)
                .finish(),
        }
    }
}

impl std::fmt::Debug for TtyFrame<'_, '_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TtyFrame::Gles(frame) => f.debug_tuple("TtyFrame::Gles").field(frame).finish(),
            TtyFrame::Vulkan(frame) => f.debug_tuple("TtyFrame::Vulkan").field(frame).finish(),
        }
    }
}

impl Texture for TtyFramebuffer<'_> {
    fn width(&self) -> u32 {
        match self {
            TtyFramebuffer::Gles(fb) => fb.width(),
            TtyFramebuffer::Vulkan(fb) => fb.width(),
        }
    }

    fn height(&self) -> u32 {
        match self {
            TtyFramebuffer::Gles(fb) => fb.height(),
            TtyFramebuffer::Vulkan(fb) => fb.height(),
        }
    }

    fn size(&self) -> Size<i32, BufferCoord> {
        match self {
            TtyFramebuffer::Gles(fb) => fb.size(),
            TtyFramebuffer::Vulkan(fb) => fb.size(),
        }
    }

    fn format(&self) -> Option<Fourcc> {
        match self {
            TtyFramebuffer::Gles(fb) => fb.format(),
            TtyFramebuffer::Vulkan(fb) => fb.format(),
        }
    }
}

impl Texture for TtyTextureMapping {
    fn width(&self) -> u32 {
        match self {
            TtyTextureMapping::Gles(mapping) => mapping.width(),
            TtyTextureMapping::Vulkan(mapping) => mapping.width(),
        }
    }

    fn height(&self) -> u32 {
        match self {
            TtyTextureMapping::Gles(mapping) => mapping.height(),
            TtyTextureMapping::Vulkan(mapping) => mapping.height(),
        }
    }

    fn format(&self) -> Option<Fourcc> {
        match self {
            TtyTextureMapping::Gles(mapping) => Texture::format(mapping),
            TtyTextureMapping::Vulkan(mapping) => Texture::format(mapping),
        }
    }
}

impl TextureMapping for TtyTextureMapping {
    fn flipped(&self) -> bool {
        match self {
            TtyTextureMapping::Gles(mapping) => mapping.flipped(),
            TtyTextureMapping::Vulkan(mapping) => mapping.flipped(),
        }
    }
}

impl<'render> RendererSuper for TtyRenderer<'render> {
    type Error = TtyRendererError;
    type TextureId = MultiTexture;
    type Framebuffer<'buffer> = TtyFramebuffer<'buffer>;
    type Frame<'frame, 'buffer>
        = TtyFrame<'render, 'frame, 'buffer>
    where
        'buffer: 'frame,
        Self: 'frame;
}

impl Renderer for TtyRenderer<'_> {
    fn context_id(&self) -> ContextId<MultiTexture> {
        match self {
            TtyRenderer::Gles(renderer) => renderer.context_id(),
            TtyRenderer::Vulkan(renderer) => renderer.context_id(),
        }
    }

    fn downscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        match self {
            TtyRenderer::Gles(renderer) => renderer.downscale_filter(filter).map_err(Into::into),
            TtyRenderer::Vulkan(renderer) => renderer.downscale_filter(filter).map_err(Into::into),
        }
    }

    fn upscale_filter(&mut self, filter: TextureFilter) -> Result<(), Self::Error> {
        match self {
            TtyRenderer::Gles(renderer) => renderer.upscale_filter(filter).map_err(Into::into),
            TtyRenderer::Vulkan(renderer) => renderer.upscale_filter(filter).map_err(Into::into),
        }
    }

    fn set_debug_flags(&mut self, flags: DebugFlags) {
        match self {
            TtyRenderer::Gles(renderer) => renderer.set_debug_flags(flags),
            TtyRenderer::Vulkan(renderer) => renderer.set_debug_flags(flags),
        }
    }

    fn debug_flags(&self) -> DebugFlags {
        match self {
            TtyRenderer::Gles(renderer) => renderer.debug_flags(),
            TtyRenderer::Vulkan(renderer) => renderer.debug_flags(),
        }
    }

    fn render<'frame, 'buffer>(
        &'frame mut self,
        framebuffer: &'frame mut Self::Framebuffer<'buffer>,
        output_size: Size<i32, Physical>,
        dst_transform: Transform,
    ) -> Result<Self::Frame<'frame, 'buffer>, Self::Error>
    where
        'buffer: 'frame,
    {
        match (self, framebuffer) {
            (TtyRenderer::Gles(renderer), TtyFramebuffer::Gles(framebuffer)) => Ok(TtyFrame::Gles(
                renderer.render(framebuffer, output_size, dst_transform)?,
            )),
            (TtyRenderer::Vulkan(renderer), TtyFramebuffer::Vulkan(framebuffer)) => Ok(
                TtyFrame::Vulkan(renderer.render(framebuffer, output_size, dst_transform)?),
            ),
            _ => unreachable!("mismatched TtyRenderer and TtyFramebuffer variants"),
        }
    }

    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        match self {
            TtyRenderer::Gles(renderer) => renderer.wait(sync).map_err(Into::into),
            TtyRenderer::Vulkan(renderer) => renderer.wait(sync).map_err(Into::into),
        }
    }

    fn cleanup_texture_cache(&mut self) -> Result<(), Self::Error> {
        match self {
            TtyRenderer::Gles(renderer) => renderer.cleanup_texture_cache().map_err(Into::into),
            TtyRenderer::Vulkan(renderer) => renderer.cleanup_texture_cache().map_err(Into::into),
        }
    }
}

impl Frame for TtyFrame<'_, '_, '_> {
    type Error = TtyRendererError;
    type TextureId = MultiTexture;

    fn context_id(&self) -> ContextId<MultiTexture> {
        match self {
            TtyFrame::Gles(frame) => frame.context_id(),
            TtyFrame::Vulkan(frame) => frame.context_id(),
        }
    }

    fn clear(
        &mut self,
        color: Color32F,
        at: &[Rectangle<i32, Physical>],
    ) -> Result<(), Self::Error> {
        match self {
            TtyFrame::Gles(frame) => frame.clear(color, at).map_err(Into::into),
            TtyFrame::Vulkan(frame) => frame.clear(color, at).map_err(Into::into),
        }
    }

    fn draw_solid(
        &mut self,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        color: Color32F,
    ) -> Result<(), Self::Error> {
        match self {
            TtyFrame::Gles(frame) => frame.draw_solid(dst, damage, color).map_err(Into::into),
            TtyFrame::Vulkan(frame) => frame.draw_solid(dst, damage, color).map_err(Into::into),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_texture_from_to(
        &mut self,
        texture: &Self::TextureId,
        src: Rectangle<f64, BufferCoord>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        src_transform: Transform,
        alpha: f32,
    ) -> Result<(), Self::Error> {
        match self {
            TtyFrame::Gles(frame) => frame
                .render_texture_from_to(
                    texture,
                    src,
                    dst,
                    damage,
                    opaque_regions,
                    src_transform,
                    alpha,
                )
                .map_err(Into::into),
            TtyFrame::Vulkan(frame) => frame
                .render_texture_from_to(
                    texture,
                    src,
                    dst,
                    damage,
                    opaque_regions,
                    src_transform,
                    alpha,
                )
                .map_err(Into::into),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_texture_at(
        &mut self,
        texture: &Self::TextureId,
        pos: Point<i32, Physical>,
        texture_scale: i32,
        output_scale: impl Into<Scale<f64>>,
        src_transform: Transform,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        alpha: f32,
    ) -> Result<(), Self::Error> {
        match self {
            TtyFrame::Gles(frame) => frame
                .render_texture_at(
                    texture,
                    pos,
                    texture_scale,
                    output_scale,
                    src_transform,
                    damage,
                    opaque_regions,
                    alpha,
                )
                .map_err(Into::into),
            TtyFrame::Vulkan(frame) => frame
                .render_texture_at(
                    texture,
                    pos,
                    texture_scale,
                    output_scale,
                    src_transform,
                    damage,
                    opaque_regions,
                    alpha,
                )
                .map_err(Into::into),
        }
    }

    fn transformation(&self) -> Transform {
        match self {
            TtyFrame::Gles(frame) => frame.transformation(),
            TtyFrame::Vulkan(frame) => frame.transformation(),
        }
    }

    fn output_size(&self) -> Size<i32, Physical> {
        match self {
            TtyFrame::Gles(frame) => frame.output_size(),
            TtyFrame::Vulkan(frame) => frame.output_size(),
        }
    }

    fn wait(&mut self, sync: &SyncPoint) -> Result<(), Self::Error> {
        match self {
            TtyFrame::Gles(frame) => frame.wait(sync).map_err(Into::into),
            TtyFrame::Vulkan(frame) => frame.wait(sync).map_err(Into::into),
        }
    }

    fn finish(self) -> Result<SyncPoint, Self::Error> {
        match self {
            TtyFrame::Gles(frame) => frame.finish().map_err(Into::into),
            TtyFrame::Vulkan(frame) => frame.finish().map_err(Into::into),
        }
    }
}

impl ImportMem for TtyRenderer<'_> {
    fn import_memory(
        &mut self,
        data: &[u8],
        format: Fourcc,
        size: Size<i32, BufferCoord>,
        flipped: bool,
    ) -> Result<Self::TextureId, Self::Error> {
        match self {
            TtyRenderer::Gles(renderer) => renderer
                .import_memory(data, format, size, flipped)
                .map_err(Into::into),
            TtyRenderer::Vulkan(renderer) => renderer
                .import_memory(data, format, size, flipped)
                .map_err(Into::into),
        }
    }

    fn update_memory(
        &mut self,
        texture: &Self::TextureId,
        data: &[u8],
        region: Rectangle<i32, BufferCoord>,
    ) -> Result<(), Self::Error> {
        match self {
            TtyRenderer::Gles(renderer) => renderer
                .update_memory(texture, data, region)
                .map_err(Into::into),
            TtyRenderer::Vulkan(renderer) => renderer
                .update_memory(texture, data, region)
                .map_err(Into::into),
        }
    }

    fn mem_formats(&self) -> Box<dyn Iterator<Item = Fourcc>> {
        match self {
            TtyRenderer::Gles(renderer) => renderer.mem_formats(),
            TtyRenderer::Vulkan(renderer) => renderer.mem_formats(),
        }
    }
}

impl ImportMemWl for TtyRenderer<'_> {
    fn import_shm_buffer(
        &mut self,
        buffer: &WlBuffer,
        surface: Option<&smithay::wayland::compositor::SurfaceData>,
        damage: &[Rectangle<i32, BufferCoord>],
    ) -> Result<Self::TextureId, Self::Error> {
        match self {
            TtyRenderer::Gles(renderer) => renderer
                .import_shm_buffer(buffer, surface, damage)
                .map_err(Into::into),
            TtyRenderer::Vulkan(renderer) => renderer
                .import_shm_buffer(buffer, surface, damage)
                .map_err(Into::into),
        }
    }
}

impl ImportDma for TtyRenderer<'_> {
    fn dmabuf_formats(&self) -> FormatSet {
        match self {
            TtyRenderer::Gles(renderer) => renderer.dmabuf_formats(),
            TtyRenderer::Vulkan(renderer) => renderer.dmabuf_formats(),
        }
    }

    fn import_dmabuf(
        &mut self,
        dmabuf: &Dmabuf,
        damage: Option<&[Rectangle<i32, BufferCoord>]>,
    ) -> Result<Self::TextureId, Self::Error> {
        match self {
            TtyRenderer::Gles(renderer) => {
                renderer.import_dmabuf(dmabuf, damage).map_err(Into::into)
            }
            TtyRenderer::Vulkan(renderer) => {
                renderer.import_dmabuf(dmabuf, damage).map_err(Into::into)
            }
        }
    }
}

impl ImportDmaWl for TtyRenderer<'_> {}

impl ImportEgl for TtyRenderer<'_> {
    fn bind_wl_display(
        &mut self,
        display: &smithay::reexports::wayland_server::DisplayHandle,
    ) -> Result<(), EglError> {
        match self {
            TtyRenderer::Gles(renderer) => renderer.bind_wl_display(display),
            // No wl_drm support on the Vulkan renderer; clients use dmabuf.
            TtyRenderer::Vulkan(_) => Err(EglError::DisplayNotSupported),
        }
    }

    fn unbind_wl_display(&mut self) {
        if let TtyRenderer::Gles(renderer) = self {
            renderer.unbind_wl_display();
        }
    }

    fn egl_reader(&self) -> Option<&EGLBufferReader> {
        match self {
            TtyRenderer::Gles(renderer) => renderer.egl_reader(),
            TtyRenderer::Vulkan(_) => None,
        }
    }

    fn import_egl_buffer(
        &mut self,
        buffer: &WlBuffer,
        surface: Option<&smithay::wayland::compositor::SurfaceData>,
        damage: &[Rectangle<i32, BufferCoord>],
    ) -> Result<Self::TextureId, Self::Error> {
        match self {
            TtyRenderer::Gles(renderer) => renderer
                .import_egl_buffer(buffer, surface, damage)
                .map_err(Into::into),
            TtyRenderer::Vulkan(_) => Err(TtyRendererError::VulkanUnsupported),
        }
    }
}

impl ExportMem for TtyRenderer<'_> {
    type TextureMapping = TtyTextureMapping;

    fn copy_framebuffer(
        &mut self,
        target: &Self::Framebuffer<'_>,
        region: Rectangle<i32, BufferCoord>,
        format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        match (self, target) {
            (TtyRenderer::Gles(renderer), TtyFramebuffer::Gles(target)) => Ok(
                TtyTextureMapping::Gles(renderer.copy_framebuffer(target, region, format)?),
            ),
            (TtyRenderer::Vulkan(renderer), TtyFramebuffer::Vulkan(target)) => Ok(
                TtyTextureMapping::Vulkan(renderer.copy_framebuffer(target, region, format)?),
            ),
            _ => unreachable!("mismatched TtyRenderer and TtyFramebuffer variants"),
        }
    }

    fn copy_texture(
        &mut self,
        texture: &Self::TextureId,
        region: Rectangle<i32, BufferCoord>,
        format: Fourcc,
    ) -> Result<Self::TextureMapping, Self::Error> {
        match self {
            TtyRenderer::Gles(renderer) => Ok(TtyTextureMapping::Gles(
                renderer.copy_texture(texture, region, format)?,
            )),
            TtyRenderer::Vulkan(renderer) => Ok(TtyTextureMapping::Vulkan(
                renderer.copy_texture(texture, region, format)?,
            )),
        }
    }

    fn can_read_texture(&mut self, texture: &Self::TextureId) -> Result<bool, Self::Error> {
        match self {
            TtyRenderer::Gles(renderer) => renderer.can_read_texture(texture).map_err(Into::into),
            TtyRenderer::Vulkan(renderer) => renderer.can_read_texture(texture).map_err(Into::into),
        }
    }

    fn map_texture<'a>(
        &mut self,
        texture_mapping: &'a Self::TextureMapping,
    ) -> Result<&'a [u8], Self::Error> {
        match (self, texture_mapping) {
            (TtyRenderer::Gles(renderer), TtyTextureMapping::Gles(mapping)) => {
                renderer.map_texture(mapping).map_err(Into::into)
            }
            (TtyRenderer::Vulkan(renderer), TtyTextureMapping::Vulkan(mapping)) => {
                renderer.map_texture(mapping).map_err(Into::into)
            }
            _ => unreachable!("mismatched TtyRenderer and TtyTextureMapping variants"),
        }
    }
}

impl Bind<Dmabuf> for TtyRenderer<'_> {
    fn bind<'a>(&mut self, target: &'a mut Dmabuf) -> Result<Self::Framebuffer<'a>, Self::Error> {
        match self {
            TtyRenderer::Gles(renderer) => Ok(TtyFramebuffer::Gles(renderer.bind(target)?)),
            TtyRenderer::Vulkan(renderer) => Ok(TtyFramebuffer::Vulkan(renderer.bind(target)?)),
        }
    }

    fn supported_formats(&self) -> Option<FormatSet> {
        match self {
            TtyRenderer::Gles(renderer) => Bind::<Dmabuf>::supported_formats(renderer),
            TtyRenderer::Vulkan(renderer) => Bind::<Dmabuf>::supported_formats(renderer),
        }
    }
}

impl Bind<GlesTexture> for TtyRenderer<'_> {
    fn bind<'a>(
        &mut self,
        target: &'a mut GlesTexture,
    ) -> Result<Self::Framebuffer<'a>, Self::Error> {
        match self {
            TtyRenderer::Gles(renderer) => Ok(TtyFramebuffer::Gles(renderer.bind(target)?)),
            TtyRenderer::Vulkan(_) => Err(TtyRendererError::VulkanUnsupported),
        }
    }
}

impl Offscreen<GlesTexture> for TtyRenderer<'_> {
    fn create_buffer(
        &mut self,
        format: Fourcc,
        size: Size<i32, BufferCoord>,
    ) -> Result<GlesTexture, Self::Error> {
        match self {
            TtyRenderer::Gles(renderer) => renderer.create_buffer(format, size).map_err(Into::into),
            TtyRenderer::Vulkan(_) => Err(TtyRendererError::VulkanUnsupported),
        }
    }
}

/// GPU manager of the TTY backend, one per selected rendering API.
pub enum TtyGpuManager {
    Gles(smithay::backend::renderer::multigpu::GpuManager<GlesApi>),
    Vulkan(smithay::backend::renderer::multigpu::GpuManager<VulkanApi>),
}

impl TtyGpuManager {
    pub fn is_vulkan(&self) -> bool {
        matches!(self, TtyGpuManager::Vulkan(_))
    }

    pub fn single_renderer(
        &mut self,
        node: &smithay::backend::drm::DrmNode,
    ) -> anyhow::Result<TtyRenderer<'_>> {
        match self {
            TtyGpuManager::Gles(gpus) => Ok(TtyRenderer::Gles(gpus.single_renderer(node)?)),
            TtyGpuManager::Vulkan(gpus) => Ok(TtyRenderer::Vulkan(gpus.single_renderer(node)?)),
        }
    }

    pub fn renderer(
        &mut self,
        render_device: &smithay::backend::drm::DrmNode,
        target_device: &smithay::backend::drm::DrmNode,
        copy_format: Fourcc,
    ) -> anyhow::Result<TtyRenderer<'_>> {
        match self {
            TtyGpuManager::Gles(gpus) => Ok(TtyRenderer::Gles(gpus.renderer(
                render_device,
                target_device,
                copy_format,
            )?)),
            TtyGpuManager::Vulkan(gpus) => Ok(TtyRenderer::Vulkan(gpus.renderer(
                render_device,
                target_device,
                copy_format,
            )?)),
        }
    }

    pub fn add_node(
        &mut self,
        node: smithay::backend::drm::DrmNode,
        gbm: smithay::backend::allocator::gbm::GbmDevice<DeviceFd>,
    ) -> anyhow::Result<()> {
        match self {
            TtyGpuManager::Gles(gpus) => gpus.as_mut().add_node(node, gbm).map_err(Into::into),
            TtyGpuManager::Vulkan(gpus) => {
                gpus.as_mut().add_node(node, gbm);
                Ok(())
            }
        }
    }

    pub fn remove_node(&mut self, node: &smithay::backend::drm::DrmNode) {
        match self {
            TtyGpuManager::Gles(gpus) => gpus.as_mut().remove_node(node),
            TtyGpuManager::Vulkan(gpus) => gpus.as_mut().remove_node(node),
        }
    }

    /// Triggers a re-enumeration of the devices.
    pub fn refresh_devices(&mut self) {
        match self {
            TtyGpuManager::Gles(gpus) => {
                let _ = gpus.devices();
            }
            TtyGpuManager::Vulkan(gpus) => {
                let _ = gpus.devices();
            }
        }
    }

    pub fn early_import(
        &mut self,
        target: smithay::backend::drm::DrmNode,
        surface: &smithay::reexports::wayland_server::protocol::wl_surface::WlSurface,
    ) -> anyhow::Result<()> {
        match self {
            TtyGpuManager::Gles(gpus) => gpus.early_import(target, surface).map_err(Into::into),
            TtyGpuManager::Vulkan(gpus) => gpus.early_import(target, surface).map_err(Into::into),
        }
    }
}
