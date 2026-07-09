//! Per-output blend space for windowed HDR support.
//!
//! An output is either SDR (electrical sRGB, the default) or HDR (the framebuffer holds
//! PQ/BT.2020 electrical values and the connector is signalled accordingly). On HDR outputs,
//! SDR content is encoded into the blend space at draw time by the shaders' `niri_blend`
//! stage; surfaces that already carry an HDR image description pass through numerically.
//!
//! Blending happens directly in PQ-encoded space. Alpha blending in an encoded space is an
//! approximation (the same class of error as regular sRGB-space blending).

use std::cell::Cell;

use smithay::backend::renderer::element::surface::WaylandSurfaceRenderElement;
use smithay::backend::renderer::element::{Element, Id, Kind, RenderElement, UnderlyingStorage};
use smithay::backend::renderer::gles::{GlesError, GlesFrame, GlesRenderer, Uniform};
use smithay::backend::renderer::utils::{CommitCounter, DamageSet, OpaqueRegions};
use smithay::backend::renderer::{Color32F, ImportAll, Renderer};
use smithay::utils::user_data::UserDataMap;
use smithay::utils::{Buffer, Physical, Point, Rectangle, Scale, Transform};
use smithay::wayland::color::management::ImageDescription;

use super::renderer::AsGlesFrame as _;
use super::shaders::Shaders;
use crate::backend::tty::{TtyFrame, TtyRenderer, TtyRendererError};

/// Default SDR reference white in cd/m² (BT.2408).
pub const DEFAULT_REFERENCE_LUMINANCE: f64 = 203.;

/// How a surface's content relates to the output blend space, derived from its committed
/// image description.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContentColor {
    /// Electrical sRGB content; encoded into the blend space on HDR outputs.
    #[default]
    Sdr,
    /// Content already encoded in the output blend space (carries an HDR image description);
    /// passes through numerically.
    BlendSpace,
    /// Windows scRGB content (linear BT.709, 1.0 = 80 cd/m²); encoded into the blend space
    /// with the fixed absolute scRGB mapping. Immune to tone mapping by definition: only
    /// clamped to the output volume, never rescaled by the SDR reference luminance.
    ///
    /// Unlike other content this is also transformed on SDR outputs (reference white anchored
    /// to display white, HDR headroom clamped away): the raw linear values would otherwise
    /// blow bright colors out to white in the framebuffer.
    Scrgb,
}

impl ContentColor {
    /// The content color of a surface with the given committed image description.
    pub fn from_description(desc: Option<ImageDescription>) -> Self {
        match desc {
            Some(desc) if desc.windows_scrgb => ContentColor::Scrgb,
            Some(desc) if desc.is_hdr() => ContentColor::BlendSpace,
            _ => ContentColor::Sdr,
        }
    }
}

/// The blend state of the frame currently being rendered, stored in the renderer's EGL user
/// data (like [`super::shaders::Shaders`]).
///
/// Shader uniform values persist in GL program objects across draws, so on HDR frames every
/// draw sets the blend uniforms from this state, and on SDR frames sets them back to zero.
#[derive(Debug, Default)]
pub struct FrameBlendState {
    hdr_pq: Cell<bool>,
    ref_lum_scale: Cell<f32>,
}

impl FrameBlendState {
    pub fn init(renderer: &mut GlesRenderer) {
        let data = renderer.egl_context().user_data();
        data.insert_if_missing(FrameBlendState::default);
    }

    fn get(renderer: &GlesRenderer) -> &Self {
        renderer
            .egl_context()
            .user_data()
            .get()
            .expect("FrameBlendState::init() must be called when creating the renderer")
    }

    /// Marks the frames rendered from now on as HDR with the given SDR reference luminance,
    /// or as SDR (`None`).
    pub fn set(renderer: &mut GlesRenderer, reference_luminance: Option<f64>) {
        let state = Self::get(renderer);
        match reference_luminance {
            Some(lum) => {
                state.hdr_pq.set(true);
                state.ref_lum_scale.set((lum / 10000.) as f32);
            }
            None => state.hdr_pq.set(false),
        }
    }

    fn values_from_frame(frame: &GlesFrame) -> (bool, f32) {
        let state: &Self = frame
            .egl_context()
            .user_data()
            .get()
            .expect("FrameBlendState::init() must be called when creating the renderer");
        (state.hdr_pq.get(), state.ref_lum_scale.get())
    }

    /// The `niri_blend` uniform values for a draw of SDR content in this frame.
    pub fn uniforms(frame: &GlesFrame) -> [Uniform<'static>; 3] {
        Self::uniforms_for_content(frame, ContentColor::Sdr)
    }

    /// The `niri_blend` uniform values for a draw in this frame; [`ContentColor::BlendSpace`]
    /// exempts content already encoded in the blend space, [`ContentColor::Scrgb`] selects the
    /// absolute scRGB encode.
    pub fn uniforms_for_content(frame: &GlesFrame, content: ContentColor) -> [Uniform<'static>; 3] {
        let (hdr_pq, scale) = Self::values_from_frame(frame);
        let apply = hdr_pq && content != ContentColor::BlendSpace;
        [
            Uniform::new("niri_hdr_pq", if apply { 1.0f32 } else { 0.0 }),
            Uniform::new("niri_ref_lum_scale", scale),
            Uniform::new(
                "niri_scrgb",
                if content == ContentColor::Scrgb {
                    1.0f32
                } else {
                    0.0
                },
            ),
        ]
    }
}

/// Configures the renderer for rendering frames in the given blend space: `Some(reference
/// luminance)` = HDR (PQ/BT.2020), `None` = SDR.
///
/// In HDR, texture draws using the default program go through the blend-space texture shader,
/// solid colors are encoded on the CPU, and niri's own shader programs read the frame blend
/// state for their `niri_blend` stage. Call with `None` after rendering the output so
/// screencasts, screenshots and other outputs stay SDR.
pub fn set_frame_blend(renderer: &mut GlesRenderer, reference_luminance: Option<f64>) {
    FrameBlendState::set(renderer, reference_luminance);

    match reference_luminance {
        Some(lum) => {
            let scale = (lum / 10000.) as f32;
            let program = Shaders::get(renderer).texture_hdr.clone();
            if let Some(program) = program {
                renderer.set_default_tex_program_override(Some((
                    program,
                    vec![
                        Uniform::new("niri_hdr_pq", 1.0f32),
                        Uniform::new("niri_ref_lum_scale", scale),
                        // Uniform values persist in the program object; reset the scRGB flag
                        // that BlendSurfaceRenderElement sets for scRGB draws.
                        Uniform::new("niri_scrgb", 0.0f32),
                    ],
                )));
            } else {
                warn!("HDR texture shader missing; SDR content will render raw");
            }
            renderer
                .set_solid_color_transform(Some(Box::new(move |color| srgb_to_pq(color, scale))));
        }
        None => {
            renderer.set_default_tex_program_override(None);
            renderer.set_solid_color_transform(None);
        }
    }
}

/// CPU counterpart of the shaders' `niri_blend`: encodes an electrical sRGB premultiplied
/// color into PQ/BT.2020 for the given SDR reference luminance scale (reference / 10000).
pub fn srgb_to_pq(color: Color32F, ref_lum_scale: f32) -> Color32F {
    let a = color.a();
    let unpremul = |c: f32| if a > 0. { c / a } else { c };

    let pq = |lin: f32| {
        const M1: f32 = 0.1593017578125;
        const M2: f32 = 78.84375;
        const C1: f32 = 0.8359375;
        const C2: f32 = 18.8515625;
        const C3: f32 = 18.6875;
        let y = lin.clamp(0., 1.).powf(M1);
        ((C1 + C2 * y) / (1. + C3 * y)).powf(M2)
    };

    let r = unpremul(color.r()).max(0.).powf(2.2);
    let g = unpremul(color.g()).max(0.).powf(2.2);
    let b = unpremul(color.b()).max(0.).powf(2.2);

    // BT.709 -> BT.2020, linear light, D65.
    let r2020 = 0.627404 * r + 0.329283 * g + 0.043313 * b;
    let g2020 = 0.069097 * r + 0.919540 * g + 0.011362 * b;
    let b2020 = 0.016391 * r + 0.088013 * g + 0.895595 * b;

    Color32F::new(
        pq(r2020 * ref_lum_scale) * a,
        pq(g2020 * ref_lum_scale) * a,
        pq(b2020 * ref_lum_scale) * a,
        a,
    )
}

/// A surface-tree render element that knows how its content relates to the output blend space
/// (from its committed image description).
///
/// For blend-space (PQ) content the frame-wide blend-space texture program is suspended around
/// the draw, so the client's PQ values pass through numerically, and underlying storage is
/// delegated so direct scanout keeps working. For scRGB content the frame program is swapped
/// for one applying the absolute scRGB encode, and direct scanout is prevented (the raw linear
/// buffer must not reach a PQ-signalled connector).
#[derive(Debug)]
pub struct BlendSurfaceRenderElement<R: Renderer> {
    inner: WaylandSurfaceRenderElement<R>,
    content: ContentColor,
}

impl<R: Renderer> BlendSurfaceRenderElement<R> {
    pub fn new(inner: WaylandSurfaceRenderElement<R>, content: ContentColor) -> Self {
        Self { inner, content }
    }

    pub fn inner(&self) -> &WaylandSurfaceRenderElement<R> {
        &self.inner
    }

    pub fn into_inner(self) -> WaylandSurfaceRenderElement<R> {
        self.inner
    }

    pub fn content(&self) -> ContentColor {
        self.content
    }
}

/// Adjusts the frame's default-texture-program override for a draw of the given content,
/// returning the previous override to restore afterwards (`None` = nothing was changed).
///
/// Blend-space content suspends the override entirely (numeric passthrough). scRGB content
/// installs the blend texture program with the `niri_scrgb` flag raised — on HDR *and* SDR
/// frames: raw linear scRGB values are meaningless on an SDR framebuffer (channels above 1.0
/// clamp to full scale, blowing bright colors out to white), so SDR frames get the
/// reference-white-anchored SDR encode from the shader instead of a passthrough.
fn adjust_tex_program_for_content(
    frame: &mut GlesFrame,
    content: ContentColor,
) -> Option<
    Option<(
        smithay::backend::renderer::gles::GlesTexProgram,
        Vec<Uniform<'static>>,
    )>,
> {
    match content {
        ContentColor::Sdr => None,
        ContentColor::BlendSpace => {
            let saved = frame.take_tex_program_override();
            saved.is_some().then_some(saved)
        }
        ContentColor::Scrgb => {
            let program = Shaders::get_from_frame(frame).texture_hdr.clone();
            let saved = frame.take_tex_program_override();
            // Prefer the frame override's program; on SDR frames (no override) fall back to
            // the blend shader directly.
            let Some(program) = saved.as_ref().map(|(p, _)| p.clone()).or(program) else {
                // Shader failed to compile at startup (already warned); render raw.
                return None;
            };
            let (hdr_pq, scale) = FrameBlendState::values_from_frame(frame);
            frame.set_tex_program_override(Some((
                program,
                vec![
                    Uniform::new("niri_hdr_pq", if hdr_pq { 1.0f32 } else { 0.0 }),
                    Uniform::new("niri_ref_lum_scale", scale),
                    Uniform::new("niri_scrgb", 1.0f32),
                ],
            )));
            Some(saved)
        }
    }
}

impl<R: Renderer + ImportAll> Element for BlendSurfaceRenderElement<R>
where
    R::TextureId: Clone + 'static,
{
    fn id(&self) -> &Id {
        self.inner.id()
    }

    fn current_commit(&self) -> CommitCounter {
        self.inner.current_commit()
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        self.inner.geometry(scale)
    }

    fn location(&self, scale: Scale<f64>) -> Point<i32, Physical> {
        self.inner.location(scale)
    }

    fn src(&self) -> Rectangle<f64, Buffer> {
        self.inner.src()
    }

    fn transform(&self) -> Transform {
        self.inner.transform()
    }

    fn damage_since(
        &self,
        scale: Scale<f64>,
        commit: Option<CommitCounter>,
    ) -> DamageSet<i32, Physical> {
        self.inner.damage_since(scale, commit)
    }

    fn opaque_regions(&self, scale: Scale<f64>) -> OpaqueRegions<i32, Physical> {
        self.inner.opaque_regions(scale)
    }

    fn alpha(&self) -> f32 {
        self.inner.alpha()
    }

    fn kind(&self) -> Kind {
        self.inner.kind()
    }
}

impl RenderElement<GlesRenderer> for BlendSurfaceRenderElement<GlesRenderer> {
    fn draw(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        cache: Option<&UserDataMap>,
    ) -> Result<(), GlesError> {
        let saved = adjust_tex_program_for_content(frame, self.content);
        let res = RenderElement::<GlesRenderer>::draw(
            &self.inner,
            frame,
            src,
            dst,
            damage,
            opaque_regions,
            cache,
        );
        if let Some(saved) = saved {
            frame.set_tex_program_override(saved);
        }
        res
    }

    fn underlying_storage(&self, renderer: &mut GlesRenderer) -> Option<UnderlyingStorage<'_>> {
        // scRGB buffers need the GL encode; their raw linear values must not be scanned out.
        if self.content == ContentColor::Scrgb {
            return None;
        }
        self.inner.underlying_storage(renderer)
    }
}

impl<'render> RenderElement<TtyRenderer<'render>>
    for BlendSurfaceRenderElement<TtyRenderer<'render>>
{
    fn draw(
        &self,
        frame: &mut TtyFrame<'render, '_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        cache: Option<&UserDataMap>,
    ) -> Result<(), TtyRendererError<'render>> {
        let gles_frame = frame.as_gles_frame();
        let saved = adjust_tex_program_for_content(gles_frame, self.content);
        let res = RenderElement::draw(&self.inner, frame, src, dst, damage, opaque_regions, cache);
        if let Some(saved) = saved {
            frame.as_gles_frame().set_tex_program_override(saved);
        }
        res
    }

    fn underlying_storage(
        &self,
        renderer: &mut TtyRenderer<'render>,
    ) -> Option<UnderlyingStorage<'_>> {
        // scRGB buffers need the GL encode; their raw linear values must not be scanned out.
        if self.content == ContentColor::Scrgb {
            return None;
        }
        self.inner.underlying_storage(renderer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_to_pq_reference_values() {
        let scale = (203. / 10000.) as f32;

        // Opaque white at reference luminance 203 cd/m²: PQ(0.0203) ≈ 0.5806.
        let white = srgb_to_pq(Color32F::new(1., 1., 1., 1.), scale);
        assert!((white.r() - 0.5806).abs() < 0.002, "got {}", white.r());
        // BT.709 white maps to BT.2020 white (rows sum to 1) => neutral stays neutral.
        assert!((white.r() - white.g()).abs() < 0.0005);
        assert!((white.g() - white.b()).abs() < 0.0005);

        // Black stays (essentially) black — PQ(0) is ~4e-7 — and alpha is preserved.
        let black = srgb_to_pq(Color32F::new(0., 0., 0., 0.5), scale);
        assert!(black.r() < 1e-6, "got {}", black.r());
        assert_eq!(black.a(), 0.5);

        // Premultiplied 50% white: unpremultiplied value is 1.0, so the encoded result is
        // the white point rescaled by alpha.
        let half = srgb_to_pq(Color32F::new(0.5, 0.5, 0.5, 0.5), scale);
        assert!((half.r() - white.r() * 0.5).abs() < 0.0005);
    }
}
