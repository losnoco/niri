use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::renderer::gles::{GlesFrame, GlesRenderer, GlesTexture};
use smithay::backend::renderer::{
    Bind, ExportMem, ImportAll, ImportMem, Offscreen, Renderer, RendererSuper, Texture,
};

use crate::backend::tty::{TtyFrame, TtyRenderer};

/// Trait with our main renderer requirements to save on the typing.
pub trait NiriRenderer:
    ImportAll
    + ImportMem
    + ExportMem
    + Bind<Dmabuf>
    + Offscreen<GlesTexture>
    + Renderer<TextureId = Self::NiriTextureId, Error = Self::NiriError>
    + AsGlesRenderer
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
    R: ImportAll + ImportMem + ExportMem + Bind<Dmabuf> + Offscreen<GlesTexture> + AsGlesRenderer,
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
