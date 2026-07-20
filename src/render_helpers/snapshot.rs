use std::cell::OnceCell;

use niri_config::BlockOutFrom;
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::element::{Kind, RenderElement};
use smithay::utils::{Logical, Physical, Point, Rectangle, Scale, Size, Transform};

use super::{render_to_encompassing_texture, ToRenderElement};
use crate::backend::tty_renderer::TtyOffscreen;
use crate::render_helpers::renderer::NiriCaptureRenderer;
use crate::render_helpers::{RenderCtx, RenderTarget};

/// Rendered-to-texture contents cache of one snapshot variant.
pub type SnapshotTexture = OnceCell<Option<(TtyOffscreen, Rectangle<i32, Physical>)>>;

/// Snapshot of a render.
#[derive(Debug)]
pub struct RenderSnapshot<C, B> {
    /// Contents for a normal render.
    ///
    /// Relative to the geometry.
    pub contents: Vec<C>,

    /// Contents that are not blocked out, but the background is blocked out.
    ///
    /// If `None` then the background doesn't have any blocked-out surfaces, and normal `contents`
    /// can be used instead.
    pub contents_with_blocked_out_bg: Option<Vec<C>>,

    /// Blocked-out contents.
    ///
    /// Relative to the geometry.
    pub blocked_out_contents: Vec<B>,

    /// Where the contents were blocked out from at the time of the snapshot.
    pub block_out_from: Option<BlockOutFrom>,

    /// Visual size of the element at the point of the snapshot.
    pub size: Size<f64, Logical>,

    /// Contents rendered into a texture (lazily, unless prebaked).
    pub texture: SnapshotTexture,

    /// Contents with blocked-out bg rendered into a texture (lazily, unless prebaked).
    pub texture_with_blocked_out_bg: SnapshotTexture,

    /// Blocked-out contents rendered into a texture (lazily, unless prebaked).
    pub blocked_out_texture: SnapshotTexture,
}

fn bake<'a, R, E>(
    cell: &'a SnapshotTexture,
    elements: impl FnOnce() -> Vec<E>,
    renderer: &mut R,
    scale: Scale<f64>,
) -> Option<&'a (TtyOffscreen, Rectangle<i32, Physical>)>
where
    R: NiriCaptureRenderer,
    R::Error: Send + Sync + 'static,
    E: RenderElement<R>,
{
    cell.get_or_init(|| {
        let _span = tracy_client::span!("RenderSnapshot::texture");

        match render_to_encompassing_texture(
            renderer,
            scale,
            Transform::Normal,
            Fourcc::Abgr8888,
            &elements(),
        ) {
            Ok((texture, _sync_point, geo)) => Some((texture, geo)),
            Err(err) => {
                warn!("error rendering snapshot contents to texture: {err:?}");
                None
            }
        }
    })
    .as_ref()
}

impl<C, B, EC, EB> RenderSnapshot<C, B>
where
    C: ToRenderElement<RenderElement = EC>,
    B: ToRenderElement<RenderElement = EB>,
{
    /// Normal contents rendered to a texture.
    pub fn contents_texture<R>(
        &self,
        renderer: &mut R,
        scale: Scale<f64>,
    ) -> Option<&(TtyOffscreen, Rectangle<i32, Physical>)>
    where
        R: NiriCaptureRenderer,
        R::Error: Send + Sync + 'static,
        EC: RenderElement<R>,
    {
        bake(
            &self.texture,
            || {
                self.contents
                    .iter()
                    .map(|baked| {
                        baked.to_render_element(Point::from((0., 0.)), scale, 1., Kind::Unspecified)
                    })
                    .collect()
            },
            renderer,
            scale,
        )
    }

    /// Contents with blocked-out bg rendered to a texture, if they differ from normal contents.
    pub fn contents_with_blocked_out_bg_texture<R>(
        &self,
        renderer: &mut R,
        scale: Scale<f64>,
    ) -> Option<&(TtyOffscreen, Rectangle<i32, Physical>)>
    where
        R: NiriCaptureRenderer,
        R::Error: Send + Sync + 'static,
        EC: RenderElement<R>,
    {
        let contents = self.contents_with_blocked_out_bg.as_ref()?;
        bake(
            &self.texture_with_blocked_out_bg,
            || {
                contents
                    .iter()
                    .map(|baked| {
                        baked.to_render_element(Point::from((0., 0.)), scale, 1., Kind::Unspecified)
                    })
                    .collect()
            },
            renderer,
            scale,
        )
    }

    /// Blocked-out contents rendered to a texture.
    pub fn blocked_out_texture<R>(
        &self,
        renderer: &mut R,
        scale: Scale<f64>,
    ) -> Option<&(TtyOffscreen, Rectangle<i32, Physical>)>
    where
        R: NiriCaptureRenderer,
        R::Error: Send + Sync + 'static,
        EB: RenderElement<R>,
    {
        bake(
            &self.blocked_out_texture,
            || {
                self.blocked_out_contents
                    .iter()
                    .map(|baked| {
                        baked.to_render_element(Point::from((0., 0.)), scale, 1., Kind::Unspecified)
                    })
                    .collect()
            },
            renderer,
            scale,
        )
    }

    pub fn texture<R>(
        &self,
        ctx: RenderCtx<R>,
        scale: Scale<f64>,
    ) -> Option<&(TtyOffscreen, Rectangle<i32, Physical>)>
    where
        R: NiriCaptureRenderer,
        R::Error: Send + Sync + 'static,
        EC: RenderElement<R>,
        EB: RenderElement<R>,
    {
        if ctx.target.should_block_out(self.block_out_from) {
            self.blocked_out_texture(ctx.renderer, scale)
        } else if ctx.target != RenderTarget::Output && self.contents_with_blocked_out_bg.is_some()
        {
            self.contents_with_blocked_out_bg_texture(ctx.renderer, scale)
        } else {
            self.contents_texture(ctx.renderer, scale)
        }
    }
}
