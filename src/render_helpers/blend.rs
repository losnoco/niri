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

use smithay::backend::drm::{Curve1DType, ScanoutColorTransform};
use smithay::backend::renderer::element::surface::WaylandSurfaceRenderElement;
use smithay::backend::renderer::element::{Element, Id, Kind, RenderElement, UnderlyingStorage};
use smithay::backend::renderer::gles::{
    GlesError, GlesFrame, GlesRenderer, Uniform, UniformName, UniformType, UniformValue,
};
use smithay::backend::renderer::utils::{CommitCounter, DamageSet, OpaqueRegions};
use smithay::backend::renderer::Color32F;
use smithay::utils::user_data::UserDataMap;
use smithay::utils::{Buffer, Physical, Point, Rectangle, Scale, Transform};
use smithay::wayland::color::management::{
    Chromaticities, ImageDescription, Primaries as CmPrimaries, PrimariesOption,
    TransferFunction as CmTransferFunction,
};

use smithay::backend::renderer::{ImportAll, Renderer};

use super::colorimetry::{self, Mat3};
use super::renderer::AsGlesFrame as _;
use super::shaders::Shaders;
use crate::backend::tty::{TtyFrame, TtyRenderer, TtyRendererError};

/// Default SDR reference white in cd/m² (BT.2408).
pub const DEFAULT_REFERENCE_LUMINANCE: f64 = 203.;

/// The container gamut of content, relative to the two spaces niri blends in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContentGamut {
    /// sRGB / BT.709 container primaries (also the assumption for surfaces without a
    /// description).
    #[default]
    Srgb,
    /// BT.2020 container primaries, i.e. the HDR blend space's own gamut.
    Bt2020,
    /// Any other container primaries: a named set like Display-P3 or raw client-provided
    /// chromaticities. Converted in linear light during composition and scanout using the
    /// same matrix (see [`colorimetry::gamut_matrix`]).
    Custom(Chromaticities),
}

impl ContentGamut {
    /// The container gamut described by an image description's primaries.
    pub fn from_primaries(primaries: &PrimariesOption) -> Self {
        let chroma = primaries
            .values
            .or_else(|| primaries.named.map(Chromaticities::from_named));
        match chroma {
            None => ContentGamut::Srgb,
            Some(c) if c == Chromaticities::from_named(CmPrimaries::Srgb) => ContentGamut::Srgb,
            Some(c) if c == Chromaticities::from_named(CmPrimaries::Bt2020) => ContentGamut::Bt2020,
            Some(c) => ContentGamut::Custom(c),
        }
    }

    /// The linear-light matrix converting this gamut into the blend space (`true` = BT.2020,
    /// `false` = BT.709/sRGB); `None` means no conversion is needed.
    ///
    /// The known pairs use the same constants as the shaders; custom chromaticities are
    /// computed (and produce the constants to ~1e-6 for the known primaries).
    fn matrix_to(self, bt2020: bool) -> Option<Mat3> {
        match (self, bt2020) {
            (ContentGamut::Srgb, false) | (ContentGamut::Bt2020, true) => None,
            (ContentGamut::Srgb, true) => Some(BT709_TO_BT2020),
            (ContentGamut::Bt2020, false) => Some(BT2020_TO_BT709),
            (ContentGamut::Custom(c), _) => {
                let target = if bt2020 {
                    CmPrimaries::Bt2020
                } else {
                    CmPrimaries::Srgb
                };
                // Degenerate chromaticities (never sent by well-behaved clients) fall back
                // to no conversion rather than producing garbage.
                colorimetry::gamut_matrix(&c, &Chromaticities::from_named(target))
                    .filter(|m| *m != colorimetry::IDENTITY)
            }
        }
    }
}

/// How a surface's content relates to the output blend space, derived from its committed
/// image description.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentColor {
    /// Electrical sRGB-transfer content; encoded into the blend space on HDR outputs.
    /// Non-sRGB container gamuts are converted in linear light after the 2.2 decode.
    Sdr {
        /// The container primaries of the content.
        gamut: ContentGamut,
    },
    /// Client content encoded as HDR PQ (or HLG, which niri passes through like PQ).
    /// BT.2020-container content passes through numerically on HDR frames; other containers
    /// are decoded, converted and re-encoded. Converted back to SDR for capture.
    HdrPq {
        /// The container primaries of the content.
        gamut: ContentGamut,
    },
    /// Extended-linear content: Windows scRGB or a parametric `ext_linear` image description
    /// (what Mesa's Vulkan WSI attaches for `VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT`
    /// swapchains). Linear light where encoded 1.0 = `max_lum` cd/m²; encoded into the blend
    /// space with the fixed absolute mapping. Immune to tone mapping by definition: only
    /// clamped to the output volume, never rescaled by the SDR reference luminance.
    ///
    /// Unlike other content this is also transformed on SDR outputs (reference white anchored
    /// to display white, HDR headroom clamped away): the raw linear values would otherwise
    /// blow bright colors out to white in the framebuffer.
    Linear {
        /// The container primaries of the content.
        gamut: ContentGamut,
        /// Luminance of encoded 1.0 in cd/m² (80 for scRGB and default ext_linear).
        max_lum: u32,
        /// Reference white luminance in cd/m² (203 for scRGB, 80 for default ext_linear).
        ref_lum: u32,
    },
}

impl Default for ContentColor {
    fn default() -> Self {
        ContentColor::Sdr {
            gamut: ContentGamut::Srgb,
        }
    }
}

impl ContentColor {
    /// The content color of a surface with the given committed image description.
    pub fn from_description(desc: Option<ImageDescription>) -> Self {
        let Some(desc) = desc else {
            return Self::default();
        };
        let gamut = ContentGamut::from_primaries(&desc.primaries);
        if desc.windows_scrgb || desc.transfer == CmTransferFunction::ExtLinear {
            let (_, max_lum, ref_lum) = desc.luminances_or_default();
            return ContentColor::Linear {
                gamut,
                max_lum: max_lum.max(1),
                ref_lum: ref_lum.max(1),
            };
        }
        // Classify on the transfer characteristic: PQ (and HLG, which niri passes through
        // the same way) is blend-space-encoded content, everything else is SDR. Notably an
        // SDR transfer in a BT.2020 container is *wide-gamut SDR*, not HDR.
        match desc.transfer {
            CmTransferFunction::St2084Pq | CmTransferFunction::Hlg => ContentColor::HdrPq { gamut },
            _ => ContentColor::Sdr { gamut },
        }
    }
}

/// BT.709 -> BT.2020 primaries in linear light (D65), row-major. Matches the shaders'
/// `to_bt2020` exactly, so plane color pipelines and gamut uniforms built from it reproduce
/// the GLES blend output ([`colorimetry::gamut_matrix`] computes the same values to ~1e-6).
const BT709_TO_BT2020: Mat3 = [
    0.627404, 0.329283, 0.043313, //
    0.069097, 0.919540, 0.011362, //
    0.016391, 0.088013, 0.895595,
];

/// BT.2020 -> BT.709 primaries in linear light (D65), matching the shaders' `to_bt709`.
const BT2020_TO_BT709: Mat3 = [
    1.660491, -0.587641, -0.072850, //
    -0.124550, 1.132900, -0.008349, //
    -0.018151, -0.100579, 1.118730,
];

/// Embeds a linear 3x3 matrix into the 3x4 (with offset column) layout of
/// `struct drm_color_ctm_3x4`.
fn mat3_to_ctm(m: Mat3) -> [f64; 12] {
    [
        m[0], m[1], m[2], 0.0, //
        m[3], m[4], m[5], 0.0, //
        m[6], m[7], m[8], 0.0,
    ]
}

/// The gamut uniforms for a draw: `niri_use_gamut` and the column-major `niri_gamut` matrix.
///
/// `enabled` distinguishes "multiply by the (possibly identity) uniform matrix" from "use the
/// shader's built-in constants" (the frame-wide default path for plain sRGB content).
fn gamut_uniforms(enabled: bool, matrix: Option<Mat3>) -> [Uniform<'static>; 2] {
    let m = matrix.unwrap_or(colorimetry::IDENTITY);
    let mut column_major = [0f32; 9];
    for row in 0..3 {
        for col in 0..3 {
            // GLES2 requires transpose = false, so transpose on the CPU.
            column_major[col * 3 + row] = m[row * 3 + col] as f32;
        }
    }
    [
        Uniform::new("niri_use_gamut", if enabled { 1.0f32 } else { 0.0 }),
        Uniform::new(
            "niri_gamut",
            UniformValue::Matrix3x3 {
                matrices: vec![column_major],
                transpose: false,
            },
        ),
    ]
}

/// The plane color transform reproducing what the blend shaders do to content of this color
/// during composition, for direct scanout via the kernel color pipeline (drm_colorop) API.
///
/// `blend_hdr` and `reference_luminance` describe the output: an HDR (PQ/BT.2020) blend space
/// with the configured SDR reference white, or an SDR output (where the shaders assume the
/// default 203 cd/m² reference, so callers should pass that).
///
/// The math mirrors `niri_blend` in `shaders/hdr.frag` stage by stage; the kernel's named
/// `PQ 125` curves use a linear scale of 1.0 = 80 cd/m², hence the /80 in the multipliers
/// (the shaders scale against 10,000 cd/m² instead, which cancels out identically).
///
/// Container gamuts are converted with the same matrices the shaders use (constants for the
/// known sRGB/BT.2020 pairs, [`colorimetry::gamut_matrix`] for custom primaries), so scanout
/// and composition stay numerically identical.
pub fn scanout_color_transform(
    content: ContentColor,
    blend_hdr: bool,
    reference_luminance: f64,
) -> ScanoutColorTransform {
    if blend_hdr {
        match content {
            // Pure 2.2 decode, scale reference white to its luminance, convert the gamut,
            // encode as PQ.
            ContentColor::Sdr { gamut } => ScanoutColorTransform {
                decode: Some(Curve1DType::Gamma22),
                multiplier: reference_luminance / 80.,
                ctm: gamut.matrix_to(true).map(mat3_to_ctm),
                encode: Some(Curve1DType::Pq125InvEotf),
            },
            // BT.2020-container PQ is already encoded in the blend space; other containers
            // are decoded, gamut-converted in linear light and re-encoded (matching the
            // shaders' niri_pq_gamut path).
            ContentColor::HdrPq { gamut } => match gamut.matrix_to(true) {
                None => ScanoutColorTransform::IDENTITY,
                Some(m) => ScanoutColorTransform {
                    decode: Some(Curve1DType::Pq125Eotf),
                    multiplier: 1.0,
                    ctm: Some(mat3_to_ctm(m)),
                    encode: Some(Curve1DType::Pq125InvEotf),
                },
            },
            // Absolute mapping: encoded 1.0 = max_lum cd/m², deliberately independent of the
            // SDR reference luminance (display-referred, never tone mapped).
            ContentColor::Linear {
                gamut,
                max_lum,
                ref_lum: _,
            } => ScanoutColorTransform {
                decode: None,
                multiplier: f64::from(max_lum) / 80.,
                ctm: gamut.matrix_to(true).map(mat3_to_ctm),
                encode: Some(Curve1DType::Pq125InvEotf),
            },
        }
    } else {
        match content {
            // SDR content on an SDR output passes through, whatever its container gamut
            // (the shaders don't convert wide-gamut SDR on SDR outputs either).
            ContentColor::Sdr { .. } => ScanoutColorTransform::IDENTITY,
            // PQ decode, anchor the reference white to display white (clamping the headroom
            // away at the encode), convert the gamut, gamma-encode.
            ContentColor::HdrPq { gamut } => ScanoutColorTransform {
                decode: Some(Curve1DType::Pq125Eotf),
                multiplier: 80. / reference_luminance,
                ctm: gamut.matrix_to(false).map(mat3_to_ctm),
                encode: Some(Curve1DType::Gamma22Inv),
            },
            // Reference white anchored to display white via the content's own reference
            // luminance, HDR headroom clamped away by the encode curve.
            ContentColor::Linear {
                gamut,
                max_lum,
                ref_lum,
            } => ScanoutColorTransform {
                decode: None,
                multiplier: f64::from(max_lum) / f64::from(ref_lum),
                ctm: gamut.matrix_to(false).map(mat3_to_ctm),
                encode: Some(Curve1DType::Gamma22Inv),
            },
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

    pub fn set_sdr_capture(renderer: &mut GlesRenderer, reference_luminance: f64) {
        let state = Self::get(renderer);
        state.hdr_pq.set(false);
        state
            .ref_lum_scale
            .set((reference_luminance / 10000.) as f32);
    }

    fn values_from_frame(frame: &GlesFrame) -> (bool, f32) {
        let state: &Self = frame
            .egl_context()
            .user_data()
            .get()
            .expect("FrameBlendState::init() must be called when creating the renderer");
        (state.hdr_pq.get(), state.ref_lum_scale.get())
    }

    pub fn is_hdr_frame(frame: &GlesFrame) -> bool {
        Self::values_from_frame(frame).0
    }

    pub fn ref_lum_scale(frame: &GlesFrame) -> f32 {
        Self::values_from_frame(frame).1
    }

    /// The compile-time declarations for every uniform the `niri_blend` stage (and the gamut
    /// uniforms) can set per draw.
    ///
    /// Every shader program that receives uniforms from [`Self::uniforms`],
    /// [`Self::uniforms_for_content`], [`Self::uniforms_for_blend_space`] or `gamut_uniforms`
    /// must include these in its uniform list: smithay rejects per-draw uniforms that were not
    /// declared when the program was compiled. Over-declaring is harmless (names the GLSL does
    /// not use resolve to location -1, which GL ignores), so programs declare the full set
    /// even when they only use a subset.
    pub fn uniform_names() -> [UniformName<'static>; 9] {
        [
            UniformName::new("niri_hdr_pq", UniformType::_1f),
            UniformName::new("niri_ref_lum_scale", UniformType::_1f),
            UniformName::new("niri_linear", UniformType::_1f),
            UniformName::new("niri_linear_scale", UniformType::_1f),
            UniformName::new("niri_linear_to_ref", UniformType::_1f),
            UniformName::new("niri_hdr_to_sdr", UniformType::_1f),
            UniformName::new("niri_pq_gamut", UniformType::_1f),
            UniformName::new("niri_use_gamut", UniformType::_1f),
            UniformName::new("niri_gamut", UniformType::Matrix3x3),
        ]
    }

    /// The `niri_blend` uniform values for content already rendered in the frame blend space.
    pub fn uniforms_for_blend_space(frame: &GlesFrame) -> Vec<Uniform<'static>> {
        let (_, scale) = Self::values_from_frame(frame);
        let mut uniforms = vec![
            Uniform::new("niri_hdr_pq", 0.0f32),
            Uniform::new("niri_ref_lum_scale", scale),
            Uniform::new("niri_linear", 0.0f32),
            Uniform::new("niri_linear_scale", 0.0f32),
            Uniform::new("niri_linear_to_ref", 0.0f32),
            Uniform::new("niri_hdr_to_sdr", 0.0f32),
            Uniform::new("niri_pq_gamut", 0.0f32),
        ];
        uniforms.extend(gamut_uniforms(false, None));
        uniforms
    }

    /// The `niri_blend` uniform values for a draw of SDR content in this frame.
    pub fn uniforms(frame: &GlesFrame) -> Vec<Uniform<'static>> {
        Self::uniforms_for_content(frame, ContentColor::default())
    }

    /// The `niri_blend` uniform values for a draw in this frame; [`ContentColor::HdrPq`]
    /// exempts BT.2020-container PQ content from SDR-to-HDR conversion (other containers are
    /// re-encoded through the gamut matrix), [`ContentColor::Linear`] selects the absolute
    /// extended-linear encode. HDR PQ content is converted back to SDR when drawn into SDR
    /// capture buffers.
    pub fn uniforms_for_content(frame: &GlesFrame, content: ContentColor) -> Vec<Uniform<'static>> {
        let (hdr_pq, scale) = Self::values_from_frame(frame);

        let is_pq = matches!(content, ContentColor::HdrPq { .. });
        let sdr_to_hdr = hdr_pq && !is_pq;
        let hdr_to_sdr = !hdr_pq && is_pq;

        // The gamut conversion into the frame's blend space, when the draw transforms the
        // content at all (SDR content on SDR frames passes through untouched).
        let gamut = match content {
            ContentColor::Sdr { .. } if !hdr_pq => None,
            ContentColor::Sdr { gamut } => Some(gamut.matrix_to(true)),
            ContentColor::HdrPq { gamut } => {
                let matrix = gamut.matrix_to(hdr_pq);
                // On HDR frames, BT.2020-container PQ passes through without any transform.
                if hdr_pq && matrix.is_none() {
                    None
                } else {
                    Some(matrix)
                }
            }
            ContentColor::Linear { gamut, .. } => Some(gamut.matrix_to(hdr_pq)),
        };
        let pq_gamut = hdr_pq && is_pq && gamut.is_some();

        let (linear, linear_scale, linear_to_ref) = match content {
            ContentColor::Linear {
                max_lum, ref_lum, ..
            } => (
                1.0f32,
                max_lum as f32 / 10000.,
                max_lum as f32 / ref_lum as f32,
            ),
            _ => (0.0, 0.0, 0.0),
        };

        let mut uniforms = vec![
            Uniform::new("niri_hdr_pq", if sdr_to_hdr { 1.0f32 } else { 0.0 }),
            Uniform::new("niri_ref_lum_scale", scale),
            Uniform::new("niri_linear", linear),
            Uniform::new("niri_linear_scale", linear_scale),
            Uniform::new("niri_linear_to_ref", linear_to_ref),
            Uniform::new("niri_hdr_to_sdr", if hdr_to_sdr { 1.0f32 } else { 0.0 }),
            Uniform::new("niri_pq_gamut", if pq_gamut { 1.0f32 } else { 0.0 }),
        ];
        uniforms.extend(gamut_uniforms(gamut.is_some(), gamut.flatten()));
        uniforms
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
                        // Uniform values persist in the program object; reset the
                        // extended-linear state that BlendSurfaceRenderElement sets for
                        // linear-content draws.
                        Uniform::new("niri_linear", 0.0f32),
                        Uniform::new("niri_linear_scale", 0.0f32),
                        Uniform::new("niri_linear_to_ref", 0.0f32),
                        Uniform::new("niri_hdr_to_sdr", 0.0f32),
                        Uniform::new("niri_pq_gamut", 0.0f32),
                        // Plain sRGB content on the default path uses the shader's built-in
                        // constants.
                        Uniform::new("niri_use_gamut", 0.0f32),
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

/// Configures the renderer for rendering into an SDR capture buffer, while preserving the
/// reference luminance needed to convert HDR content back to SDR.
pub fn set_sdr_capture_blend(renderer: &mut GlesRenderer, reference_luminance: f64) {
    FrameBlendState::set_sdr_capture(renderer, reference_luminance);
    renderer.set_default_tex_program_override(None);
    renderer.set_solid_color_transform(None);
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

/// CPU encode of a premultiplied ARGB8888 buffer (little-endian, so B,G,R,A bytes) from
/// electrical sRGB into PQ/BT.2020, for the cursor plane on HDR outputs: its contents bypass
/// the renderer, so the conversion the blend shaders would do runs here instead.
///
/// Runs only when the cursor *image* changes (cursor movement reuses the buffer), so the
/// per-pixel `powf` cost is acceptable.
pub fn srgb_to_pq_argb8888(data: &mut [u8], stride: u32, size: (u32, u32), ref_lum_scale: f32) {
    let (width, height) = size;
    for row in 0..height as usize {
        let start = row * stride as usize;
        let row_data = &mut data[start..start + width as usize * 4];
        for px in row_data.chunks_exact_mut(4) {
            let color = Color32F::new(
                f32::from(px[2]) / 255.,
                f32::from(px[1]) / 255.,
                f32::from(px[0]) / 255.,
                f32::from(px[3]) / 255.,
            );
            let color = srgb_to_pq(color, ref_lum_scale);
            px[2] = (color.r() * 255.).round().clamp(0., 255.) as u8;
            px[1] = (color.g() * 255.).round().clamp(0., 255.) as u8;
            px[0] = (color.b() * 255.).round().clamp(0., 255.) as u8;
        }
    }
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
/// HDR PQ client content suspends the override on HDR frames (numeric passthrough) and installs
/// the HDR-to-SDR texture program on SDR capture frames. Extended-linear content installs the
/// blend texture program with the `niri_linear` state set — on HDR *and* SDR frames: raw
/// extended-linear values are meaningless on an SDR framebuffer (channels above 1.0 clamp to full
/// scale, blowing bright colors out to white), so SDR frames get the reference-white-anchored SDR
/// encode from the shader instead of a passthrough.
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
        // Plain sRGB SDR content uses the frame's default path (the blend-space program on
        // HDR frames, no transform on SDR frames).
        ContentColor::Sdr {
            gamut: ContentGamut::Srgb,
        } => None,
        // Wide-gamut SDR content needs the gamut uniform on HDR frames; on SDR frames it
        // passes through like any SDR content.
        ContentColor::Sdr { .. } => {
            if !FrameBlendState::is_hdr_frame(frame) {
                return None;
            }
            let program = Shaders::get_from_frame(frame).texture_hdr.clone();
            let saved = frame.take_tex_program_override();
            let Some(program) = saved.as_ref().map(|(p, _)| p.clone()).or(program) else {
                // Shader failed to compile at startup (already warned); render raw.
                return None;
            };
            let uniforms = FrameBlendState::uniforms_for_content(frame, content);
            frame.set_tex_program_override(Some((program, uniforms)));
            Some(saved)
        }
        ContentColor::HdrPq { gamut } => {
            let saved = frame.take_tex_program_override();
            if FrameBlendState::is_hdr_frame(frame) {
                match gamut.matrix_to(true) {
                    // BT.2020-container PQ passes through numerically.
                    None => saved.is_some().then_some(saved),
                    // Other containers are decoded, converted and re-encoded.
                    Some(_) => {
                        let program = Shaders::get_from_frame(frame).texture_hdr.clone();
                        let Some(program) = saved.as_ref().map(|(p, _)| p.clone()).or(program)
                        else {
                            return saved.is_some().then_some(saved);
                        };
                        let uniforms = FrameBlendState::uniforms_for_content(frame, content);
                        frame.set_tex_program_override(Some((program, uniforms)));
                        Some(saved)
                    }
                }
            } else if let Some(program) = Shaders::get_from_frame(frame).texture_hdr_to_sdr.clone()
            {
                let ref_lum_scale = FrameBlendState::ref_lum_scale(frame);
                let mut uniforms = vec![Uniform::new("niri_ref_lum_scale", ref_lum_scale)];
                // Convert the container gamut to BT.709; the uniform (rather than the
                // shader's built-in 2020 constant) also covers non-2020 containers.
                uniforms.extend(gamut_uniforms(true, gamut.matrix_to(false)));
                frame.override_default_tex_program(program, uniforms);
                Some(saved)
            } else {
                warn!("HDR-to-SDR texture shader missing; HDR capture will render raw");
                Some(saved)
            }
        }
        ContentColor::Linear { .. } => {
            let program = Shaders::get_from_frame(frame).texture_hdr.clone();
            let saved = frame.take_tex_program_override();
            // Prefer the frame override's program; on SDR frames (no override) fall back to
            // the blend shader directly.
            let Some(program) = saved.as_ref().map(|(p, _)| p.clone()).or(program) else {
                // Shader failed to compile at startup (already warned); render raw.
                return None;
            };
            let uniforms = FrameBlendState::uniforms_for_content(frame, content);
            frame.set_tex_program_override(Some((program, uniforms)));
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
        // Raw buffer values must never reach a connector whose signal differs from them
        // (e.g. an extended-linear buffer on any output, or an SDR buffer on a PQ output);
        // the TTY backend guards this by handing the DrmCompositor a per-element
        // ScanoutColorTransform for every window surface, which either programs the
        // conversion into the plane's color pipeline or keeps the element composited.
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
        // Raw buffer values must never reach a connector whose signal differs from them;
        // see the GlesRenderer impl above for how the TTY backend guards this.
        self.inner.underlying_storage(renderer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_color_classification() {
        use smithay::wayland::color::management::PrimariesOption;

        assert_eq!(
            ContentColor::from_description(None),
            ContentColor::default()
        );
        assert_eq!(
            ContentColor::from_description(Some(ImageDescription::SRGB)),
            ContentColor::Sdr {
                gamut: ContentGamut::Srgb
            }
        );

        // Windows scRGB: 1.0 = 80 cd/m² with a 203 cd/m² reference white.
        assert_eq!(
            ContentColor::from_description(Some(ImageDescription::WINDOWS_SCRGB)),
            ContentColor::Linear {
                gamut: ContentGamut::Srgb,
                max_lum: 80,
                ref_lum: 203,
            }
        );

        // A parametric ext_linear + sRGB description (Mesa's WSI mapping for
        // VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) with default luminances.
        let mesa_scrgb = ImageDescription {
            transfer: CmTransferFunction::ExtLinear,
            ..ImageDescription::SRGB
        };
        assert_eq!(
            ContentColor::from_description(Some(mesa_scrgb)),
            ContentColor::Linear {
                gamut: ContentGamut::Srgb,
                max_lum: 80,
                ref_lum: 80,
            }
        );

        // BT.2020 linear content skips the 709 -> 2020 conversion.
        let bt2020_linear = ImageDescription {
            transfer: CmTransferFunction::ExtLinear,
            primaries: PrimariesOption {
                named: Some(CmPrimaries::Bt2020),
                values: None,
            },
            ..ImageDescription::SRGB
        };
        assert!(matches!(
            ContentColor::from_description(Some(bt2020_linear)),
            ContentColor::Linear {
                gamut: ContentGamut::Bt2020,
                ..
            }
        ));

        // PQ content passes through the blend space numerically.
        let pq = ImageDescription {
            transfer: CmTransferFunction::St2084Pq,
            primaries: PrimariesOption {
                named: Some(CmPrimaries::Bt2020),
                values: None,
            },
            ..ImageDescription::SRGB
        };
        assert_eq!(
            ContentColor::from_description(Some(pq)),
            ContentColor::HdrPq {
                gamut: ContentGamut::Bt2020
            }
        );
        assert_eq!(
            ContentColor::from_description(Some(ImageDescription::WINDOWS_BT2100)),
            ContentColor::HdrPq {
                gamut: ContentGamut::Bt2020
            }
        );

        // PQ in a Display-P3 container keeps its custom gamut for conversion.
        let p3_chroma = Chromaticities::from_named(CmPrimaries::DisplayP3);
        let pq_p3 = ImageDescription {
            transfer: CmTransferFunction::St2084Pq,
            primaries: PrimariesOption {
                named: Some(CmPrimaries::DisplayP3),
                values: None,
            },
            ..ImageDescription::SRGB
        };
        assert_eq!(
            ContentColor::from_description(Some(pq_p3)),
            ContentColor::HdrPq {
                gamut: ContentGamut::Custom(p3_chroma)
            }
        );

        // Raw chromaticities equal to a known named set normalize to the named gamut.
        let raw_srgb = ImageDescription {
            primaries: PrimariesOption {
                named: None,
                values: Some(Chromaticities::from_named(CmPrimaries::Srgb)),
            },
            ..ImageDescription::SRGB
        };
        assert_eq!(
            ContentColor::from_description(Some(raw_srgb)),
            ContentColor::Sdr {
                gamut: ContentGamut::Srgb
            }
        );

        // An SDR transfer in a BT.2020 container is wide-gamut SDR, not HDR.
        let wide_sdr = ImageDescription {
            primaries: PrimariesOption {
                named: Some(CmPrimaries::Bt2020),
                values: None,
            },
            ..ImageDescription::SRGB
        };
        assert_eq!(
            ContentColor::from_description(Some(wide_sdr)),
            ContentColor::Sdr {
                gamut: ContentGamut::Bt2020
            }
        );
    }

    #[test]
    fn scanout_transforms_mirror_the_shaders() {
        use smithay::backend::drm::Curve1DType;

        let srgb = ContentGamut::Srgb;
        let bt2020 = ContentGamut::Bt2020;

        // SDR on an HDR output: gamma 2.2 decode, reference white at 203 cd/m² = a gain of
        // 203/80 on the PQ-125 linear scale, 709 -> 2020, PQ encode.
        let sdr = scanout_color_transform(ContentColor::Sdr { gamut: srgb }, true, 203.);
        assert_eq!(sdr.decode, Some(Curve1DType::Gamma22));
        assert!((sdr.multiplier - 2.5375).abs() < 1e-9);
        assert_eq!(sdr.ctm, Some(mat3_to_ctm(BT709_TO_BT2020)));
        assert_eq!(sdr.encode, Some(Curve1DType::Pq125InvEotf));

        // BT.2020-container PQ on an HDR output passes through numerically, like the shaders.
        assert!(
            scanout_color_transform(ContentColor::HdrPq { gamut: bt2020 }, true, 203.)
                .is_identity()
        );
        // ... and anything on an SDR output that is already SDR needs no transform.
        assert!(
            scanout_color_transform(ContentColor::Sdr { gamut: srgb }, false, 203.).is_identity()
        );

        // PQ in a Display-P3 container on an HDR output: decode, convert, re-encode.
        let p3 = ContentGamut::Custom(Chromaticities::from_named(CmPrimaries::DisplayP3));
        let pq_p3 = scanout_color_transform(ContentColor::HdrPq { gamut: p3 }, true, 203.);
        assert_eq!(pq_p3.decode, Some(Curve1DType::Pq125Eotf));
        assert_eq!(pq_p3.multiplier, 1.0);
        assert_eq!(pq_p3.encode, Some(Curve1DType::Pq125InvEotf));
        let ctm = pq_p3.ctm.unwrap();
        // Display-P3 and BT.2020 share the D65 white point: rows sum to 1.
        for row in 0..3 {
            let sum: f64 = ctm[row * 4..row * 4 + 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "row {row} sums to {sum}");
            assert_eq!(ctm[row * 4 + 3], 0.0);
        }

        // Windows scRGB on an HDR output: already linear at 1.0 = 80 cd/m², which is
        // exactly the PQ-125 scale, so only the gamut conversion and the PQ encode remain.
        let scrgb = scanout_color_transform(
            ContentColor::Linear {
                gamut: srgb,
                max_lum: 80,
                ref_lum: 203,
            },
            true,
            203.,
        );
        assert_eq!(scrgb.decode, None);
        assert_eq!(scrgb.multiplier, 1.0);
        assert_eq!(scrgb.ctm, Some(mat3_to_ctm(BT709_TO_BT2020)));
        assert_eq!(scrgb.encode, Some(Curve1DType::Pq125InvEotf));

        // BT.2020 extended-linear content skips the gamut conversion.
        let linear_2020 = scanout_color_transform(
            ContentColor::Linear {
                gamut: bt2020,
                max_lum: 80,
                ref_lum: 203,
            },
            true,
            203.,
        );
        assert_eq!(linear_2020.ctm, None);

        // PQ content on an SDR output: decode, anchor 203 cd/m² to display white,
        // 2020 -> 709, gamma encode (clamping the headroom away).
        let pq_on_sdr = scanout_color_transform(ContentColor::HdrPq { gamut: bt2020 }, false, 203.);
        assert_eq!(pq_on_sdr.decode, Some(Curve1DType::Pq125Eotf));
        assert!((pq_on_sdr.multiplier - 80. / 203.).abs() < 1e-9);
        assert_eq!(pq_on_sdr.ctm, Some(mat3_to_ctm(BT2020_TO_BT709)));
        assert_eq!(pq_on_sdr.encode, Some(Curve1DType::Gamma22Inv));

        // scRGB on an SDR output: reference white (203) anchored to display white.
        let scrgb_on_sdr = scanout_color_transform(
            ContentColor::Linear {
                gamut: srgb,
                max_lum: 80,
                ref_lum: 203,
            },
            false,
            203.,
        );
        assert_eq!(scrgb_on_sdr.decode, None);
        assert!((scrgb_on_sdr.multiplier - 80. / 203.).abs() < 1e-9);
        assert_eq!(scrgb_on_sdr.ctm, None);
        assert_eq!(scrgb_on_sdr.encode, Some(Curve1DType::Gamma22Inv));

        // Rows of the built-in matrices sum to ~1 (white maps to white).
        for row in 0..3 {
            let sum: f64 = BT709_TO_BT2020[row * 3..row * 3 + 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row {row} sums to {sum}");
            let sum: f64 = BT2020_TO_BT709[row * 3..row * 3 + 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row {row} sums to {sum}");
        }
    }

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
