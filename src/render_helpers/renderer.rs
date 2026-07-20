use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::renderer::gles::{GlesFrame, GlesRenderer, GlesTexture};
#[allow(unused_imports)]
use smithay::backend::renderer::Frame;
use smithay::backend::renderer::{
    Bind, ExportMem, ImportAll, ImportMem, Offscreen, Renderer, RendererSuper, Texture,
};

use crate::backend::tty::{TtyFrame, TtyRenderer};
use crate::backend::tty_renderer::TtyOffscreen;

/// Trait with our main renderer requirements to save on the typing.
pub trait NiriRenderer:
    ImportAll
    + ImportMem
    + ExportMem
    + Bind<Dmabuf>
    + Offscreen<GlesTexture>
    + Renderer<TextureId = Self::NiriTextureId, Error = Self::NiriError>
    + AsGlesRenderer
    + AsVulkanRenderer
    + HasOffscreen
    + Offscreen<<Self as HasOffscreen>::Offscreen>
    + Bind<<Self as HasOffscreen>::Offscreen>
    + crate::render_helpers::blend::CaptureBlend
{
    // Associated types to work around the instability of associated type bounds.
    type NiriTextureId: Texture + Clone + Send + 'static;
    type NiriError: std::error::Error
        + Send
        + Sync
        + From<<GlesRenderer as RendererSuper>::Error>
        + 'static;
}

impl<R> NiriRenderer for R
where
    R: ImportAll
        + ImportMem
        + ExportMem
        + Bind<Dmabuf>
        + Offscreen<GlesTexture>
        + AsGlesRenderer
        + AsVulkanRenderer
        + HasOffscreen
        + Offscreen<<R as HasOffscreen>::Offscreen>
        + Bind<<R as HasOffscreen>::Offscreen>
        + crate::render_helpers::blend::CaptureBlend,
    R::TextureId: Texture + Clone + Send + 'static,
    R::Error:
        std::error::Error + Send + Sync + From<<GlesRenderer as RendererSuper>::Error> + 'static,
{
    type NiriTextureId = R::TextureId;
    type NiriError = R::Error;
}

/// Trait for getting the underlying `GlesRenderer`, if any.
///
/// Returns `None` for the Vulkan renderer; GLES-specific functionality (custom shaders,
/// offscreen effects) degrades gracefully in that case.
pub trait AsGlesRenderer {
    fn as_gles_renderer(&mut self) -> Option<&mut GlesRenderer>;
}

impl AsGlesRenderer for GlesRenderer {
    fn as_gles_renderer(&mut self) -> Option<&mut GlesRenderer> {
        Some(self)
    }
}

impl AsGlesRenderer for TtyRenderer<'_> {
    fn as_gles_renderer(&mut self) -> Option<&mut GlesRenderer> {
        match self {
            TtyRenderer::Gles(renderer) => Some(renderer.as_mut()),
            TtyRenderer::Vulkan(_) => None,
        }
    }
}

/// Trait for getting the underlying `GlesFrame`, if any.
pub trait AsGlesFrame<'frame, 'buffer>
where
    Self: 'frame,
{
    fn as_gles_frame(&mut self) -> Option<&mut GlesFrame<'frame, 'buffer>>;
}

impl<'frame, 'buffer> AsGlesFrame<'frame, 'buffer> for GlesFrame<'frame, 'buffer> {
    fn as_gles_frame(&mut self) -> Option<&mut GlesFrame<'frame, 'buffer>> {
        Some(self)
    }
}

impl<'frame, 'buffer> AsGlesFrame<'frame, 'buffer> for TtyFrame<'_, 'frame, 'buffer> {
    fn as_gles_frame(&mut self) -> Option<&mut GlesFrame<'frame, 'buffer>> {
        match self {
            TtyFrame::Gles(frame) => Some(frame.as_mut()),
            TtyFrame::Vulkan(_) => None,
        }
    }
}

/// Trait for getting the underlying `VulkanRenderer`, if any.
pub trait AsVulkanRenderer {
    fn as_vulkan_renderer(
        &mut self,
    ) -> Option<&mut smithay::backend::renderer::vulkan::VulkanRenderer>;
}

impl AsVulkanRenderer for GlesRenderer {
    fn as_vulkan_renderer(
        &mut self,
    ) -> Option<&mut smithay::backend::renderer::vulkan::VulkanRenderer> {
        None
    }
}

impl AsVulkanRenderer for TtyRenderer<'_> {
    fn as_vulkan_renderer(
        &mut self,
    ) -> Option<&mut smithay::backend::renderer::vulkan::VulkanRenderer> {
        match self {
            TtyRenderer::Gles(_) => None,
            TtyRenderer::Vulkan(renderer) => Some(renderer.as_mut()),
        }
    }
}

/// The offscreen render target texture type of a renderer.
pub trait HasOffscreen: Renderer {
    type Offscreen: Texture + Clone + Send + 'static;

    /// Wraps the renderer's offscreen texture into the renderer-agnostic enum.
    fn wrap_offscreen(texture: Self::Offscreen) -> TtyOffscreen;
    /// Unwraps the renderer-agnostic enum back into this renderer's texture type.
    fn unwrap_offscreen(texture: &mut TtyOffscreen) -> Option<&mut Self::Offscreen>;
}

impl HasOffscreen for GlesRenderer {
    type Offscreen = GlesTexture;

    fn wrap_offscreen(texture: GlesTexture) -> TtyOffscreen {
        TtyOffscreen::Gles(texture)
    }

    fn unwrap_offscreen(texture: &mut TtyOffscreen) -> Option<&mut GlesTexture> {
        match texture {
            TtyOffscreen::Gles(texture) => Some(texture),
            TtyOffscreen::Vulkan(_) => None,
        }
    }
}

impl HasOffscreen for TtyRenderer<'_> {
    type Offscreen = crate::backend::tty_renderer::TtyOffscreen;

    fn wrap_offscreen(texture: TtyOffscreen) -> TtyOffscreen {
        texture
    }

    fn unwrap_offscreen(texture: &mut TtyOffscreen) -> Option<&mut TtyOffscreen> {
        Some(texture)
    }
}

/// Renderer bounds needed by the generic capture helpers (screenshots, screencopy,
/// screencasting): offscreen render targets of the renderer's own texture type plus the
/// SDR-capture blend control.
pub trait NiriCaptureRenderer:
    NiriRenderer
    + HasOffscreen
    + Offscreen<<Self as HasOffscreen>::Offscreen>
    + Bind<<Self as HasOffscreen>::Offscreen>
    + crate::render_helpers::blend::CaptureBlend
{
}

impl<R> NiriCaptureRenderer for R where
    R: NiriRenderer
        + HasOffscreen
        + Offscreen<<R as HasOffscreen>::Offscreen>
        + Bind<<R as HasOffscreen>::Offscreen>
        + crate::render_helpers::blend::CaptureBlend
{
}
