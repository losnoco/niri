//! On-screen strip of minimized windows.
//!
//! Minimized windows are taken out of the layout entirely, so without something like this they
//! have no on-screen representation at all. This draws a small thumbnail of each one in a corner
//! of every output, and clicking a thumbnail restores that window.
//!
//! The strip is deliberately stateless: geometry is recomputed from the layout on every render and
//! on every hit test, by the same function, so the two can't drift apart. Minimized windows are
//! suspended and stop drawing, so a thumbnail shows the last frame before minimizing.

use smithay::backend::renderer::element::utils::{
    Relocate, RelocateRenderElement, RescaleRenderElement,
};
use smithay::backend::renderer::element::{Kind, RenderElement};
use smithay::output::Output;
use smithay::utils::{Logical, Point, Rectangle, Scale, Size};

use crate::layout::{LayoutElement as _, LayoutElementRenderElement};
use crate::niri::Niri;
use crate::niri_render_elements;
use crate::render_helpers::renderer::NiriRenderer;
use crate::render_helpers::solid_color::{SolidColorBuffer, SolidColorRenderElement};
use crate::render_helpers::RenderCtx;
use crate::utils::{output_size, round_logical_in_physical};
use crate::window::mapped::MappedId;

niri_render_elements! {
    MinimizedStripRenderElement<R> => {
        Thumbnail = RelocateRenderElement<RescaleRenderElement<LayoutElementRenderElement<R>>>,
        Backdrop = SolidColorRenderElement,
    }
}

/// Backdrop drawn behind each thumbnail so that dark windows stay visible.
const BACKDROP_COLOR: [f32; 4] = [0., 0., 0., 0.4];

/// How much of the backdrop sticks out around a thumbnail.
const BACKDROP_PADDING: f64 = 4.;

/// Placement of one minimized window's thumbnail.
pub struct ThumbnailGeo {
    pub id: MappedId,
    /// Where the thumbnail goes, in output-local logical coordinates.
    pub geo: Rectangle<f64, Logical>,
    /// Logical size of the window itself, which the thumbnail is a scaled-down copy of.
    pub window_size: Size<f64, Logical>,
}

/// Computes where each minimized window's thumbnail goes on this output.
///
/// Returns an empty vec when the strip is off, when nothing is minimized, or when the output is
/// too small to fit even one thumbnail. Thumbnails that would overflow the output are dropped
/// rather than squeezed, so the ones that are shown stay legible.
pub fn thumbnail_geometry(niri: &Niri, output: &Output) -> Vec<ThumbnailGeo> {
    let config = niri.config.borrow();
    let config = &config.minimized_windows;

    if config.off {
        return Vec::new();
    }

    let windows: Vec<_> = niri
        .layout
        .minimized_windows()
        .map(|mapped| (mapped.id(), mapped.size().to_f64()))
        .collect();

    layout_thumbnails(
        &windows,
        config,
        output_size(output),
        output.current_scale().fractional_scale(),
    )
}

/// Lays out the strip. Split out from [`thumbnail_geometry`] so it can be tested without a
/// compositor.
fn layout_thumbnails(
    windows: &[(MappedId, Size<f64, Logical>)],
    config: &niri_config::MinimizedWindows,
    output_size: Size<f64, Logical>,
    scale: f64,
) -> Vec<ThumbnailGeo> {
    let round = |logical: f64| round_logical_in_physical(scale, logical);

    let size = round(config.size);
    let gaps = round(config.gaps);

    // The row is `size` tall regardless of the aspect ratios in it, so thumbnails of different
    // shapes still line up along a common baseline.
    let row_y = if config.position.is_top() {
        gaps
    } else {
        output_size.h - gaps - size
    };

    let mut placed: Vec<ThumbnailGeo> = Vec::new();
    let mut used_w = 0.;

    for &(id, window_size) in windows {
        if window_size.w <= 0. || window_size.h <= 0. {
            continue;
        }

        // Fit the window into a size x size box without distorting it.
        let factor = f64::min(size / window_size.w, size / window_size.h);
        let thumb_size = Size::from((round(window_size.w * factor), round(window_size.h * factor)));

        let advance = thumb_size.w + if placed.is_empty() { 0. } else { gaps };
        if used_w + advance + gaps * 2. > output_size.w {
            // Out of room. Better to show fewer thumbnails than unreadable ones.
            break;
        }
        used_w += advance;

        placed.push(ThumbnailGeo {
            id,
            // x is filled in below, once the total width is known.
            geo: Rectangle::new(
                Point::from((0., row_y + (size - thumb_size.h) / 2.)),
                thumb_size,
            ),
            window_size,
        });
    }

    if placed.is_empty() {
        return placed;
    }

    // Right-hand corners grow leftwards, so the strip has to be laid out from its total width.
    let mut x = if config.position.is_left() {
        gaps
    } else {
        output_size.w - gaps - used_w
    };

    for thumb in &mut placed {
        thumb.geo.loc.x = x;
        x += thumb.geo.size.w + gaps;
    }

    placed
}

/// Returns the minimized window whose thumbnail is under this output-local point, if any.
pub fn window_under(
    niri: &Niri,
    output: &Output,
    pos_within_output: Point<f64, Logical>,
) -> Option<MappedId> {
    thumbnail_geometry(niri, output)
        .into_iter()
        // Later thumbnails are drawn on top, so hit-test in reverse.
        .rev()
        .find(|thumb| thumb.geo.to_f64().contains(pos_within_output))
        .map(|thumb| thumb.id)
}

/// Draws the strip on top of everything else on this output.
pub fn render_output<R: NiriRenderer>(
    niri: &Niri,
    output: &Output,
    mut ctx: RenderCtx<R>,
    push: &mut dyn FnMut(MinimizedStripRenderElement<R>),
) where
    MinimizedStripRenderElement<R>: RenderElement<R>,
{
    let thumbs = thumbnail_geometry(niri, output);
    if thumbs.is_empty() {
        return;
    }

    let _span = tracy_client::span!("minimized_strip::render_output");

    let scale = output.current_scale().fractional_scale();
    let s = Scale::from(scale);

    // Keep one buffer per thumbnail so that they can differ in size without thrashing. Buffers
    // live in Niri because rendering only gets &self.
    let mut all_buffers = niri.minimized_strip_buffers.borrow_mut();
    let backdrop_buffers = all_buffers.entry(output.clone()).or_default();
    backdrop_buffers.resize_with(thumbs.len(), SolidColorBuffer::default);

    // Render element lists are front-to-back: whatever is pushed first ends up on top. So all
    // thumbnails go in first, and the backdrops follow to sit behind them.
    for thumb in &thumbs {
        let Some(mapped) = niri
            .layout
            .minimized_windows()
            .find(|mapped| mapped.id() == thumb.id)
        else {
            continue;
        };

        let thumb_scale = Scale {
            x: thumb.geo.size.w / thumb.window_size.w,
            y: thumb.geo.size.h / thumb.window_size.h,
        };
        let loc = thumb.geo.loc.to_physical_precise_round(scale);

        // Render the window at the origin, then scale it down and move it into place. Same shape
        // as the MRU switcher's thumbnails.
        //
        // FIXME: this could use mipmaps, for that it should be rendered through an offscreen.
        mapped.render_normal(ctx.r(), Point::from((0., 0.)), s, 1., &mut |elem| {
            let elem = RescaleRenderElement::from_element(elem, Point::from((0, 0)), thumb_scale);
            let elem = RelocateRenderElement::from_element(elem, loc, Relocate::Relative);
            push(MinimizedStripRenderElement::Thumbnail(elem));
        });
    }

    for (thumb, buffer) in thumbs.iter().zip(backdrop_buffers.iter_mut()) {
        let backdrop_geo = Rectangle::new(
            thumb.geo.loc - Point::from((BACKDROP_PADDING, BACKDROP_PADDING)),
            thumb.geo.size + Size::from((BACKDROP_PADDING * 2., BACKDROP_PADDING * 2.)),
        );
        buffer.resize(backdrop_geo.size);
        buffer.set_color(BACKDROP_COLOR);
        push(MinimizedStripRenderElement::Backdrop(
            SolidColorRenderElement::from_buffer(buffer, backdrop_geo.loc, 1., Kind::Unspecified),
        ));
    }
}

#[cfg(test)]
mod tests {
    use niri_config::{MinimizedPosition, MinimizedWindows};

    use super::*;

    fn config(position: MinimizedPosition) -> MinimizedWindows {
        MinimizedWindows {
            off: false,
            position,
            size: 100.,
            gaps: 10.,
        }
    }

    fn windows(sizes: &[(f64, f64)]) -> Vec<(MappedId, Size<f64, Logical>)> {
        sizes
            .iter()
            .map(|&(w, h)| (MappedId::next(), Size::from((w, h))))
            .collect()
    }

    const OUTPUT: (f64, f64) = (1000., 500.);

    #[test]
    fn thumbnails_preserve_aspect_ratio() {
        // A 200x100 window is twice as wide as tall, so it fits the 100 box by width.
        let wins = windows(&[(200., 100.)]);
        let placed = layout_thumbnails(
            &wins,
            &config(MinimizedPosition::BottomLeft),
            OUTPUT.into(),
            1.,
        );

        assert_eq!(placed.len(), 1);
        assert_eq!(placed[0].geo.size.w, 100.);
        assert_eq!(placed[0].geo.size.h, 50.);
    }

    #[test]
    fn bottom_left_starts_at_the_left_edge() {
        let wins = windows(&[(100., 100.), (100., 100.)]);
        let placed = layout_thumbnails(
            &wins,
            &config(MinimizedPosition::BottomLeft),
            OUTPUT.into(),
            1.,
        );

        assert_eq!(placed[0].geo.loc.x, 10.);
        assert_eq!(placed[1].geo.loc.x, 120.); // 10 + 100 + 10 gap
                                               // Row sits one gap above the bottom edge.
        assert_eq!(placed[0].geo.loc.y, 500. - 10. - 100.);
    }

    #[test]
    fn bottom_right_grows_leftwards_and_ends_at_the_right_edge() {
        let wins = windows(&[(100., 100.), (100., 100.)]);
        let placed = layout_thumbnails(
            &wins,
            &config(MinimizedPosition::BottomRight),
            OUTPUT.into(),
            1.,
        );

        // Total width is 100 + 10 + 100 = 210, so the strip starts at 1000 - 10 - 210.
        assert_eq!(placed[0].geo.loc.x, 780.);
        assert_eq!(placed[1].geo.loc.x, 890.);
        // Last thumbnail's right edge is one gap from the output's right edge.
        assert_eq!(placed[1].geo.loc.x + placed[1].geo.size.w, 990.);
    }

    #[test]
    fn top_positions_sit_at_the_top() {
        let wins = windows(&[(100., 100.)]);
        let placed = layout_thumbnails(
            &wins,
            &config(MinimizedPosition::TopLeft),
            OUTPUT.into(),
            1.,
        );

        assert_eq!(placed[0].geo.loc.y, 10.);
    }

    #[test]
    fn overflow_is_dropped_rather_than_squeezed() {
        // Each is 100 wide with 10 gaps, plus 10 margin each side: 8 fit, the 9th does not.
        let wins = windows(&[(100., 100.); 20]);
        let placed = layout_thumbnails(
            &wins,
            &config(MinimizedPosition::BottomLeft),
            OUTPUT.into(),
            1.,
        );

        assert!(
            placed.len() < 20,
            "some thumbnails should have been dropped"
        );
        let last = placed.last().unwrap();
        assert!(
            last.geo.loc.x + last.geo.size.w <= 1000.,
            "shown thumbnails must stay on the output"
        );
        // Whatever is shown keeps full size.
        assert!(placed.iter().all(|t| t.geo.size.w == 100.));
    }

    #[test]
    fn degenerate_window_sizes_are_skipped() {
        let wins = windows(&[(0., 0.), (100., 100.)]);
        let placed = layout_thumbnails(
            &wins,
            &config(MinimizedPosition::BottomLeft),
            OUTPUT.into(),
            1.,
        );

        assert_eq!(placed.len(), 1);
        assert_eq!(placed[0].id, wins[1].0);
    }

    #[test]
    fn shorter_thumbnails_are_centered_in_the_row() {
        // 200x100 becomes 100x50, so it gets 25 of slack above and below inside the 100 row.
        let wins = windows(&[(200., 100.)]);
        let placed = layout_thumbnails(
            &wins,
            &config(MinimizedPosition::BottomLeft),
            OUTPUT.into(),
            1.,
        );

        assert_eq!(placed[0].geo.loc.y, 500. - 10. - 100. + 25.);
    }
}
