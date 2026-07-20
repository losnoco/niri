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
use smithay::backend::renderer::vulkan::{ColorBlendParams, VulkanFrame, VulkanRenderer};
use smithay::backend::renderer::{Color32F, ImportAll, Renderer};
use smithay::utils::user_data::UserDataMap;
use smithay::utils::{Buffer, Physical, Point, Rectangle, Scale, Transform};
use smithay::wayland::color::management::{
    Chromaticities, ImageDescription, Primaries as CmPrimaries, PrimariesOption,
    TransferFunction as CmTransferFunction,
};

use super::colorimetry::{self, Mat3};
use super::renderer::AsGlesFrame as _;
use super::shaders::Shaders;
use crate::backend::tty::{TtyFrame, TtyRenderer, TtyRendererError};

/// Default SDR reference white in cd/m² (BT.2408).
pub const DEFAULT_REFERENCE_LUMINANCE: f64 = 203.;

/// Relative headroom below which tone mapping is skipped and clipping suffices (KWin uses the
/// same epsilon).
const TONEMAP_ETA: f64 = 1.001;

/// Whether tone mapping is enabled at all; `NIRI_DISABLE_TONEMAPPING=1` restores the previous
/// clip-at-the-output-volume behavior (mirroring `KWIN_DISABLE_TONEMAPPING`).
fn tonemapping_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NIRI_DISABLE_TONEMAPPING").is_none_or(|v| v != "1"))
}

/// Whether content peaking at `content_max` cd/m² needs tone mapping into an output peaking
/// at `output_max` cd/m² (with a little headroom slack, like KWin, so clipping handles
/// near-misses).
fn tonemap_needed(content_max: Option<u32>, output_max: f64) -> bool {
    tonemapping_enabled()
        && output_max > 0.
        && content_max.is_some_and(|max| f64::from(max) > output_max * TONEMAP_ETA)
}

/// The parameter `v` of the modified Reinhard curve `f(l) = l * (1 + l*v) / (1 + l)` (KWin's
/// `ColorTonemapper`), solved so that `f(input_range) = output_range`, with the ranges
/// relative to the reference white. The curve satisfies `f(0) = 0` and `f(x) <= x`, and with
/// `input_range -> infinity`, `f(1) = 0.5`: reference-white content dims by at most half.
fn tonemap_curve_v(input_range: f64, output_range: f64) -> f64 {
    (output_range * (1. + input_range) - input_range) / (input_range * input_range)
}

/// The peak luminance in cd/m² an output can represent, for tone mapping decisions.
///
/// HDR outputs peak at the sink's EDID desired-content-max-luminance; without that
/// information the PQ ceiling is used, which disables tone mapping (the sink then maps
/// internally, as before). SDR outputs peak at the reference white: everything brighter is
/// HDR headroom the signal cannot carry.
pub fn output_peak_luminance(blend_hdr: bool, reference_luminance: f64, edid_max: u16) -> f64 {
    if blend_hdr {
        if edid_max > 0 {
            f64::from(edid_max)
        } else {
            10000.
        }
    } else {
        reference_luminance
    }
}

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
    ///
    /// Content whose peak luminance exceeds what the output can represent is tone mapped
    /// during composition (a modified Reinhard curve on the ICtCp intensity, like KWin):
    /// on SDR frames the headroom above reference white is compressed instead of clipped,
    /// and on HDR outputs the content peak is compressed into the sink's EDID peak.
    HdrPq {
        /// The container primaries of the content.
        gamut: ContentGamut,
        /// Peak luminance of the content in cd/m² (max CLL, else the mastering display max,
        /// else the transfer function default — 10,000 for PQ, 1,000 for HLG), or `None` for
        /// display-referred content (the Windows-BT.2100 stimulus encoding), which is exempt
        /// from tone mapping by definition and only ever clamped.
        max_lum: Option<u32>,
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
            CmTransferFunction::St2084Pq | CmTransferFunction::Hlg => ContentColor::HdrPq {
                gamut,
                // Windows-BT.2100 content is display-referred for a PQ-mode screen and thus
                // exempt from tone mapping, like Windows-scRGB.
                max_lum: (!desc.windows_bt2100).then(|| desc.max_luminance().max(1)),
            },
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
/// default 203 cd/m² reference, so callers should pass that). `peak_luminance` is the
/// output's peak in cd/m² (see [`output_peak_luminance`]): content the shaders would *tone
/// map* into that peak returns `None`, since the parametric decode/multiply/matrix/encode
/// pipeline cannot express the tone mapping curve — the element must stay composited so the
/// shaders can apply it.
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
    peak_luminance: f64,
) -> Option<ScanoutColorTransform> {
    if let ContentColor::HdrPq { max_lum, .. } = content {
        if tonemap_needed(max_lum, peak_luminance) {
            return None;
        }
    }

    Some(if blend_hdr {
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
            ContentColor::HdrPq { gamut, .. } => match gamut.matrix_to(true) {
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
            // away at the encode), convert the gamut, gamma-encode. Only reachable with tone
            // mapping disabled (or display-referred content): PQ content above the reference
            // white otherwise tone maps and returns `None` above.
            ContentColor::HdrPq { gamut, .. } => ScanoutColorTransform {
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
    })
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
    /// The output's peak luminance in cd/m² (SDR frames: the reference white), for tone
    /// mapping decisions; 0 = unknown, tone mapping disabled.
    max_luminance: Cell<f32>,
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

    /// Marks the frames rendered from now on as HDR with the given SDR reference luminance
    /// and output peak luminance (both cd/m²), or as SDR (`None`).
    ///
    /// SDR frames tone map HDR content into the reference white the shaders assume for them
    /// (the BT.2408 default, matching the scanout transforms for SDR outputs).
    pub fn set(renderer: &mut GlesRenderer, blend: Option<(f64, f64)>) {
        Self::get(renderer).set_values(blend);
    }

    /// Sets the frame blend values on this state directly.
    pub fn set_values(&self, blend: Option<(f64, f64)>) {
        let state = self;
        match blend {
            Some((ref_lum, max_lum)) => {
                state.hdr_pq.set(true);
                state.ref_lum_scale.set((ref_lum / 10000.) as f32);
                state.max_luminance.set(max_lum as f32);
            }
            None => {
                state.hdr_pq.set(false);
                state
                    .ref_lum_scale
                    .set((DEFAULT_REFERENCE_LUMINANCE / 10000.) as f32);
                state.max_luminance.set(DEFAULT_REFERENCE_LUMINANCE as f32);
            }
        }
    }

    pub fn set_sdr_capture(renderer: &mut GlesRenderer, reference_luminance: f64) {
        Self::get(renderer).set_sdr_capture_values(reference_luminance);
    }

    /// Sets the SDR capture values on this state directly.
    pub fn set_sdr_capture_values(&self, reference_luminance: f64) {
        self.hdr_pq.set(false);
        self.ref_lum_scale
            .set((reference_luminance / 10000.) as f32);
        self.max_luminance.set(reference_luminance as f32);
    }

    fn values_from_frame(frame: &GlesFrame) -> (bool, f32, f32) {
        let state: &Self = frame
            .egl_context()
            .user_data()
            .get()
            .expect("FrameBlendState::init() must be called when creating the renderer");
        (
            state.hdr_pq.get(),
            state.ref_lum_scale.get(),
            state.max_luminance.get(),
        )
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
    pub fn uniform_names() -> [UniformName<'static>; 13] {
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
            UniformName::new("niri_tonemap", UniformType::_1f),
            UniformName::new("niri_tm_v", UniformType::_1f),
            UniformName::new("niri_tm_ref_scale", UniformType::_1f),
            UniformName::new("niri_tm_out_scale", UniformType::_1f),
        ]
    }

    /// The tone mapping uniforms for a draw of content peaking at `max_in` cd/m² into an
    /// output peaking at `max_out` cd/m² with reference white `ref_lum` cd/m², or the
    /// disabled state when tone mapping does not apply.
    ///
    /// The curve parameter `v` is derived like KWin's `ColorTonemapper` so that
    /// `f(input_range) = output_range` (ranges relative to the reference white); the shader
    /// applies `f(l) = l * (1 + l*v) / (1 + l)` to the ICtCp intensity.
    fn tonemap_uniforms(
        enabled: bool,
        max_in: f64,
        ref_lum: f64,
        max_out: f64,
    ) -> [Uniform<'static>; 4] {
        let (tonemap, v, ref_scale, out_scale) =
            Self::tonemap_values(enabled, max_in, ref_lum, max_out);
        [
            Uniform::new("niri_tonemap", tonemap),
            Uniform::new("niri_tm_v", v),
            Uniform::new("niri_tm_ref_scale", ref_scale),
            Uniform::new("niri_tm_out_scale", out_scale),
        ]
    }

    fn tonemap_values(
        enabled: bool,
        max_in: f64,
        ref_lum: f64,
        max_out: f64,
    ) -> (f32, f32, f32, f32) {
        if !enabled || ref_lum <= 0. || max_out <= 0. {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let input_range = max_in / ref_lum;
        let output_range = max_out / ref_lum;
        let v = tonemap_curve_v(input_range, output_range);
        (
            1.0,
            v as f32,
            (ref_lum / 10000.) as f32,
            (max_out / 10000.) as f32,
        )
    }

    /// The `niri_blend` uniform values for content already rendered in the frame blend space.
    pub fn uniforms_for_blend_space(frame: &GlesFrame) -> Vec<Uniform<'static>> {
        let (_, scale, _) = Self::values_from_frame(frame);
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
        uniforms.extend(Self::tonemap_uniforms(false, 0., 0., 0.));
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
        let (hdr_pq, scale, max_luminance) = Self::values_from_frame(frame);
        Self::uniforms_for_content_values(hdr_pq, scale, max_luminance, content)
    }

    /// The `niri_blend` parameter block for a draw in a Vulkan frame; the Vulkan counterpart
    /// of [`Self::uniforms_for_content`].
    pub fn vulkan_params_for_content(
        frame: &VulkanFrame,
        content: ContentColor,
    ) -> ColorBlendParams {
        let (hdr_pq, scale, max_luminance) = frame
            .user_data()
            .get::<Self>()
            .map(|state| {
                (
                    state.hdr_pq.get(),
                    state.ref_lum_scale.get(),
                    state.max_luminance.get(),
                )
            })
            .unwrap_or((
                false,
                (DEFAULT_REFERENCE_LUMINANCE / 10000.) as f32,
                DEFAULT_REFERENCE_LUMINANCE as f32,
            ));
        Self::values_for_content(hdr_pq, scale, max_luminance, content)
    }

    fn uniforms_for_content_values(
        hdr_pq: bool,
        scale: f32,
        max_luminance: f32,
        content: ContentColor,
    ) -> Vec<Uniform<'static>> {
        let values = Self::values_for_content(hdr_pq, scale, max_luminance, content);
        let uniforms = vec![
            Uniform::new("niri_hdr_pq", values.hdr_pq),
            Uniform::new("niri_ref_lum_scale", values.ref_lum_scale),
            Uniform::new("niri_linear", values.linear),
            Uniform::new("niri_linear_scale", values.linear_scale),
            Uniform::new("niri_linear_to_ref", values.linear_to_ref),
            Uniform::new("niri_hdr_to_sdr", values.hdr_to_sdr),
            Uniform::new("niri_pq_gamut", values.pq_gamut),
            Uniform::new("niri_use_gamut", values.use_gamut),
            Uniform::new(
                "niri_gamut",
                UniformValue::Matrix3x3 {
                    matrices: vec![values.gamut],
                    transpose: false,
                },
            ),
            Uniform::new("niri_tonemap", values.tonemap),
            Uniform::new("niri_tm_v", values.tm_v),
            Uniform::new("niri_tm_ref_scale", values.tm_ref_scale),
            Uniform::new("niri_tm_out_scale", values.tm_out_scale),
        ];
        uniforms
    }

    /// Computes the raw `niri_blend` parameter block for a draw.
    fn values_for_content(
        hdr_pq: bool,
        scale: f32,
        max_luminance: f32,
        content: ContentColor,
    ) -> ColorBlendParams {
        let is_pq = matches!(content, ContentColor::HdrPq { .. });
        let sdr_to_hdr = hdr_pq && !is_pq;
        let hdr_to_sdr = !hdr_pq && is_pq;

        // Tone mapping applies to PQ content whose peak exceeds the frame's output peak:
        // the reference white of the frame anchors the curve, so on SDR frames (peak =
        // reference white) the headroom compresses into the SDR range.
        let tonemap = match content {
            ContentColor::HdrPq { max_lum, .. } => {
                tonemap_needed(max_lum, f64::from(max_luminance))
            }
            _ => false,
        };

        // The gamut conversion into the frame's blend space, when the draw transforms the
        // content at all (SDR content on SDR frames passes through untouched).
        let gamut = match content {
            ContentColor::Sdr { .. } if !hdr_pq => None,
            ContentColor::Sdr { gamut } => Some(gamut.matrix_to(true)),
            ContentColor::HdrPq { gamut, .. } => {
                let matrix = gamut.matrix_to(hdr_pq);
                // On HDR frames, BT.2020-container PQ passes through without any transform
                // (unless it needs tone mapping, which forces the decode/re-encode path).
                if hdr_pq && matrix.is_none() && !tonemap {
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

        let use_gamut = gamut.is_some();
        let m = gamut.flatten().unwrap_or(colorimetry::IDENTITY);
        let mut column_major = [0f32; 9];
        for row in 0..3 {
            for col in 0..3 {
                column_major[col * 3 + row] = m[row * 3 + col] as f32;
            }
        }

        let max_in = match content {
            ContentColor::HdrPq { max_lum, .. } => f64::from(max_lum.unwrap_or(0)),
            _ => 0.,
        };
        let (tm, tm_v, tm_ref_scale, tm_out_scale) = Self::tonemap_values(
            tonemap,
            max_in,
            f64::from(scale) * 10000.,
            f64::from(max_luminance),
        );

        ColorBlendParams {
            hdr_pq: if sdr_to_hdr { 1.0 } else { 0.0 },
            ref_lum_scale: scale,
            linear,
            linear_scale,
            linear_to_ref,
            hdr_to_sdr: if hdr_to_sdr { 1.0 } else { 0.0 },
            pq_gamut: if pq_gamut { 1.0 } else { 0.0 },
            use_gamut: if use_gamut { 1.0 } else { 0.0 },
            gamut: column_major,
            tonemap: tm,
            tm_v,
            tm_ref_scale,
            tm_out_scale,
        }
    }
}

/// Configures the renderer for rendering frames in the given blend space: `Some((reference
/// luminance, output peak luminance))` = HDR (PQ/BT.2020), `None` = SDR.
///
/// In HDR, texture draws using the default program go through the blend-space texture shader,
/// solid colors are encoded on the CPU, and niri's own shader programs read the frame blend
/// state for their `niri_blend` stage. PQ content brighter than the output peak is tone
/// mapped at draw time. Call with `None` after rendering the output so screencasts,
/// screenshots and other outputs stay SDR.
pub fn set_frame_blend(renderer: &mut GlesRenderer, blend: Option<(f64, f64)>) {
    FrameBlendState::set(renderer, blend);

    match blend {
        Some((lum, _)) => {
            let scale = (lum / 10000.) as f32;
            let program = Shaders::get(renderer).and_then(|s| s.texture_hdr.clone());
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
                        // ... and never needs tone mapping (reference white fits the output).
                        Uniform::new("niri_tonemap", 0.0f32),
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

/// [`set_frame_blend`] over the TTY backend renderer.
///
/// On the Vulkan renderer, HDR frames install default [`ColorBlendParams`] performing the
/// sRGB-to-PQ encode for every texture draw and a CPU transform for solid colors, mirroring
/// the GLES default-program override.
pub fn set_frame_blend_tty(renderer: &mut TtyRenderer, blend: Option<(f64, f64)>) {
    match renderer {
        TtyRenderer::Gles(multi) => set_frame_blend(multi.as_mut(), blend),
        TtyRenderer::Vulkan(multi) => {
            let vk: &mut VulkanRenderer = multi.as_mut();
            vk.user_data().insert_if_missing(FrameBlendState::default);
            vk.user_data()
                .get::<FrameBlendState>()
                .unwrap()
                .set_values(blend);

            match blend {
                Some((ref_lum, _)) => {
                    let scale = (ref_lum / 10000.) as f32;
                    vk.set_default_color_params(Some(ColorBlendParams {
                        hdr_pq: 1.0,
                        ref_lum_scale: scale,
                        ..Default::default()
                    }));
                    vk.set_solid_color_transform(Some(Box::new(move |color| {
                        srgb_to_pq(color, scale)
                    })));
                }
                None => {
                    vk.set_default_color_params(None);
                    vk.set_solid_color_transform(None);
                }
            }
        }
    }
}

/// Configures the renderer for rendering into an SDR capture buffer, while preserving the
/// reference luminance needed to convert HDR content back to SDR.
pub fn set_sdr_capture_blend<R: CaptureBlend + ?Sized>(renderer: &mut R, reference_luminance: f64) {
    renderer.set_sdr_capture_blend(reference_luminance);
}

/// Configuring a renderer for rendering into SDR capture buffers.
pub trait CaptureBlend {
    /// See [`set_sdr_capture_blend`].
    fn set_sdr_capture_blend(&mut self, reference_luminance: f64);
}

impl CaptureBlend for GlesRenderer {
    fn set_sdr_capture_blend(&mut self, reference_luminance: f64) {
        FrameBlendState::set_sdr_capture(self, reference_luminance);
        self.set_default_tex_program_override(None);
        self.set_solid_color_transform(None);
    }
}

impl CaptureBlend for TtyRenderer<'_> {
    fn set_sdr_capture_blend(&mut self, reference_luminance: f64) {
        match self {
            TtyRenderer::Gles(multi) => {
                CaptureBlend::set_sdr_capture_blend(multi.as_mut(), reference_luminance)
            }
            TtyRenderer::Vulkan(multi) => {
                let vk: &mut VulkanRenderer = multi.as_mut();
                vk.user_data().insert_if_missing(FrameBlendState::default);
                vk.user_data()
                    .get::<FrameBlendState>()
                    .unwrap()
                    .set_sdr_capture_values(reference_luminance);
                vk.set_default_color_params(None);
                vk.set_solid_color_transform(None);
            }
        }
    }
}

/// The ST 2084 PQ inverse EOTF over clamped linear light.
fn pq_encode(lin: f32) -> f32 {
    const M1: f32 = 0.1593017578125;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.8359375;
    const C2: f32 = 18.8515625;
    const C3: f32 = 18.6875;
    let y = lin.clamp(0., 1.).powf(M1);
    ((C1 + C2 * y) / (1. + C3 * y)).powf(M2)
}

/// BT.709 -> BT.2020, linear light, D65.
fn bt709_to_bt2020(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        0.627404 * r + 0.329283 * g + 0.043313 * b,
        0.069097 * r + 0.919540 * g + 0.011362 * b,
        0.016391 * r + 0.088013 * g + 0.895595 * b,
    )
}

/// CPU counterpart of the shaders' `niri_blend`: encodes an electrical sRGB premultiplied
/// color into PQ/BT.2020 for the given SDR reference luminance scale (reference / 10000).
pub fn srgb_to_pq(color: Color32F, ref_lum_scale: f32) -> Color32F {
    let a = color.a();
    let unpremul = |c: f32| if a > 0. { c / a } else { c };

    let r = unpremul(color.r()).max(0.).powf(2.2);
    let g = unpremul(color.g()).max(0.).powf(2.2);
    let b = unpremul(color.b()).max(0.).powf(2.2);

    let (r2020, g2020, b2020) = bt709_to_bt2020(r, g, b);

    Color32F::new(
        pq_encode(r2020 * ref_lum_scale) * a,
        pq_encode(g2020 * ref_lum_scale) * a,
        pq_encode(b2020 * ref_lum_scale) * a,
        a,
    )
}

/// LUT-accelerated CPU encode of a premultiplied ARGB8888 buffer (little-endian, so B,G,R,A
/// bytes) from electrical sRGB into PQ/BT.2020, for the cursor plane on HDR outputs: its
/// contents bypass the renderer, so the conversion the blend shaders would do runs here
/// instead.
///
/// This runs inside `render_frame` every time the cursor *image* changes, which an animated
/// cursor does many times per second; the per-pixel `powf` version of this encode took
/// 12-46 ms per change on a 256x256 cursor plane and dropped frames whenever the pointer
/// showed an animated or frequently-changing cursor.
pub struct SrgbToPqEncoder {
    ref_lum_scale: f32,
    /// sRGB EOTF sampled at every 8-bit electrical value (exact for opaque pixels,
    /// interpolated after unpremultiplication otherwise).
    eotf: [f32; 256],
    /// `pq_encode` with the reference-luminance scale folded in, sampled on a quartic
    /// domain so the near-black region (where PQ is steepest) gets most of the samples:
    /// `pq[i] = pq_encode((i / (N - 1))^4 * ref_lum_scale)`.
    pq: Box<[f32; Self::PQ_SAMPLES]>,
}

impl SrgbToPqEncoder {
    const PQ_SAMPLES: usize = 4096;

    pub fn new(ref_lum_scale: f32) -> Self {
        let mut eotf = [0f32; 256];
        for (i, v) in eotf.iter_mut().enumerate() {
            *v = (i as f32 / 255.).powf(2.2);
        }

        let mut pq = Box::new([0f32; Self::PQ_SAMPLES]);
        for (i, v) in pq.iter_mut().enumerate() {
            let t = i as f32 / (Self::PQ_SAMPLES - 1) as f32;
            let lin = (t * t) * (t * t);
            *v = pq_encode(lin * ref_lum_scale);
        }

        Self {
            ref_lum_scale,
            eotf,
            pq,
        }
    }

    /// sRGB EOTF for an unpremultiplied electrical value in [0, 1] (or above, for buffers
    /// violating the premultiplication invariant).
    fn eotf_lookup(&self, x: f32) -> f32 {
        if x >= 1. {
            return x.powf(2.2);
        }
        let pos = x.max(0.) * 255.;
        let i = pos as usize;
        let frac = pos - i as f32;
        self.eotf[i] + (self.eotf[i + 1] - self.eotf[i]) * frac
    }

    /// PQ encode of a linear-light value in [0, 1], pre-scaled by the reference luminance.
    fn pq_lookup(&self, lin: f32) -> f32 {
        if lin >= 1. {
            // Out-of-range linear light from invalid premultiplied input; match the exact
            // path, whose clamp only applies after the reference-luminance scale.
            return pq_encode(lin * self.ref_lum_scale);
        }
        let t = lin.max(0.).sqrt().sqrt();
        let pos = t * (Self::PQ_SAMPLES - 1) as f32;
        let i = pos as usize;
        let frac = pos - i as f32;
        self.pq[i] + (self.pq[i + 1] - self.pq[i]) * frac
    }

    pub fn apply(&self, data: &mut [u8], stride: u32, size: (u32, u32)) {
        let _span = tracy_client::span!("SrgbToPqEncoder::apply");

        let (width, height) = size;
        let row_len = width as usize * 4;
        // The data is typically a mapping of the cursor buffer object, which may be
        // uncached; do the per-byte work in a regular allocation and copy back.
        let mut row_buf = vec![0u8; row_len];
        for row in 0..height as usize {
            let start = row * stride as usize;
            let row_data = &mut data[start..start + row_len];
            row_buf.copy_from_slice(row_data);
            for px in row_buf.chunks_exact_mut(4) {
                let a = px[3];
                if a == 0 {
                    // Premultiplied: fully transparent pixels encode to zero.
                    px[0] = 0;
                    px[1] = 0;
                    px[2] = 0;
                    continue;
                }

                let (r, g, b) = if a == 255 {
                    (
                        self.eotf[px[2] as usize],
                        self.eotf[px[1] as usize],
                        self.eotf[px[0] as usize],
                    )
                } else {
                    let a_f = f32::from(a);
                    (
                        self.eotf_lookup(f32::from(px[2]) / a_f),
                        self.eotf_lookup(f32::from(px[1]) / a_f),
                        self.eotf_lookup(f32::from(px[0]) / a_f),
                    )
                };

                let (r2020, g2020, b2020) = bt709_to_bt2020(r, g, b);

                let a_scale = f32::from(a);
                px[2] = (self.pq_lookup(r2020) * a_scale).round().clamp(0., 255.) as u8;
                px[1] = (self.pq_lookup(g2020) * a_scale).round().clamp(0., 255.) as u8;
                px[0] = (self.pq_lookup(b2020) * a_scale).round().clamp(0., 255.) as u8;
            }
            row_data.copy_from_slice(&row_buf);
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
        ContentColor::HdrPq { gamut, max_lum } => {
            let saved = frame.take_tex_program_override();
            if FrameBlendState::is_hdr_frame(frame) {
                let (_, _, frame_max_lum) = FrameBlendState::values_from_frame(frame);
                let tonemap = tonemap_needed(max_lum, f64::from(frame_max_lum));
                match gamut.matrix_to(true) {
                    // BT.2020-container PQ within the output's peak passes through
                    // numerically.
                    None if !tonemap => saved.is_some().then_some(saved),
                    // Other containers are decoded, converted and re-encoded; content
                    // brighter than the output peak is additionally tone mapped.
                    _ => {
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
                let (_, ref_lum_scale, frame_max_lum) = FrameBlendState::values_from_frame(frame);
                let mut uniforms = vec![Uniform::new("niri_ref_lum_scale", ref_lum_scale)];
                // Convert the container gamut to BT.709; the uniform (rather than the
                // shader's built-in 2020 constant) also covers non-2020 containers.
                uniforms.extend(gamut_uniforms(true, gamut.matrix_to(false)));
                // Compress the headroom above the SDR reference white instead of clipping
                // it (uniform values persist in the program, so always set the state).
                uniforms.extend(FrameBlendState::tonemap_uniforms(
                    tonemap_needed(max_lum, f64::from(frame_max_lum)),
                    f64::from(max_lum.unwrap_or(0)),
                    f64::from(ref_lum_scale) * 10000.,
                    f64::from(frame_max_lum),
                ));
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
        let saved = frame
            .as_gles_frame()
            .and_then(|gles_frame| adjust_tex_program_for_content(gles_frame, self.content));
        let saved_vk = if let crate::backend::tty::TtyFrame::Vulkan(multi) = &mut *frame {
            let vk_frame: &mut VulkanFrame = multi.as_mut();
            let params = FrameBlendState::vulkan_params_for_content(vk_frame, self.content);
            let prev = vk_frame.take_color_params_override();
            vk_frame.set_color_params_override(Some(params));
            Some(prev)
        } else {
            None
        };
        let res = RenderElement::draw(&self.inner, frame, src, dst, damage, opaque_regions, cache);
        if let Some(saved) = saved {
            if let Some(gles_frame) = frame.as_gles_frame() {
                gles_frame.set_tex_program_override(saved);
            }
        }
        if let Some(prev) = saved_vk {
            if let crate::backend::tty::TtyFrame::Vulkan(multi) = &mut *frame {
                let vk_frame: &mut VulkanFrame = multi.as_mut();
                vk_frame.set_color_params_override(prev);
            }
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
                gamut: ContentGamut::Bt2020,
                // No explicit luminance information: the PQ ceiling.
                max_lum: Some(10_000),
            }
        );
        // Windows-BT.2100 content is display-referred and exempt from tone mapping.
        assert_eq!(
            ContentColor::from_description(Some(ImageDescription::WINDOWS_BT2100)),
            ContentColor::HdrPq {
                gamut: ContentGamut::Bt2020,
                max_lum: None,
            }
        );

        // Explicit content light level information tightens the tone mapping bound.
        let pq_mastered = ImageDescription {
            max_cll: Some(1_000),
            mastering_luminance: Some((0, 4_000)),
            ..pq
        };
        assert_eq!(
            ContentColor::from_description(Some(pq_mastered)),
            ContentColor::HdrPq {
                gamut: ContentGamut::Bt2020,
                max_lum: Some(1_000),
            }
        );
        let pq_mastered_no_cll = ImageDescription {
            mastering_luminance: Some((0, 4_000)),
            ..pq
        };
        assert_eq!(
            ContentColor::from_description(Some(pq_mastered_no_cll)),
            ContentColor::HdrPq {
                gamut: ContentGamut::Bt2020,
                max_lum: Some(4_000),
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
                gamut: ContentGamut::Custom(p3_chroma),
                max_lum: Some(10_000),
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

        // An output without EDID luminance information: the PQ ceiling, no tone mapping.
        let peak = 10_000.;

        // SDR on an HDR output: gamma 2.2 decode, reference white at 203 cd/m² = a gain of
        // 203/80 on the PQ-125 linear scale, 709 -> 2020, PQ encode.
        let sdr =
            scanout_color_transform(ContentColor::Sdr { gamut: srgb }, true, 203., peak).unwrap();
        assert_eq!(sdr.decode, Some(Curve1DType::Gamma22));
        assert!((sdr.multiplier - 2.5375).abs() < 1e-9);
        assert_eq!(sdr.ctm, Some(mat3_to_ctm(BT709_TO_BT2020)));
        assert_eq!(sdr.encode, Some(Curve1DType::Pq125InvEotf));

        // BT.2020-container PQ on an HDR output passes through numerically, like the shaders.
        let pq_2020 = ContentColor::HdrPq {
            gamut: bt2020,
            max_lum: Some(10_000),
        };
        assert!(scanout_color_transform(pq_2020, true, 203., peak)
            .unwrap()
            .is_identity());
        // ... and anything on an SDR output that is already SDR needs no transform.
        assert!(
            scanout_color_transform(ContentColor::Sdr { gamut: srgb }, false, 203., 203.)
                .unwrap()
                .is_identity()
        );

        // PQ in a Display-P3 container on an HDR output: decode, convert, re-encode.
        let p3 = ContentGamut::Custom(Chromaticities::from_named(CmPrimaries::DisplayP3));
        let pq_p3 = scanout_color_transform(
            ContentColor::HdrPq {
                gamut: p3,
                max_lum: Some(10_000),
            },
            true,
            203.,
            peak,
        )
        .unwrap();
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
            peak,
        )
        .unwrap();
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
            peak,
        )
        .unwrap();
        assert_eq!(linear_2020.ctm, None);

        // Display-referred PQ content on an SDR output (exempt from tone mapping): decode,
        // anchor 203 cd/m² to display white, 2020 -> 709, gamma encode (clamping the
        // headroom away).
        let pq_on_sdr = scanout_color_transform(
            ContentColor::HdrPq {
                gamut: bt2020,
                max_lum: None,
            },
            false,
            203.,
            203.,
        )
        .unwrap();
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
            203.,
        )
        .unwrap();
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
    fn tone_mapped_content_is_denied_scanout() {
        let bt2020 = ContentGamut::Bt2020;
        let pq = |max_lum| ContentColor::HdrPq {
            gamut: bt2020,
            max_lum,
        };

        // PQ content brighter than the sink's EDID peak is tone mapped during composition;
        // no parametric plane pipeline can express the curve, so scanout must be denied.
        assert_eq!(
            scanout_color_transform(pq(Some(10_000)), true, 203., 800.),
            None
        );
        assert_eq!(
            scanout_color_transform(pq(Some(1_000)), true, 203., 800.),
            None
        );
        // Content within the peak scans out.
        assert!(scanout_color_transform(pq(Some(750)), true, 203., 800.)
            .unwrap()
            .is_identity());
        // A tiny overshoot is clipping territory, not tone mapping (the eta slack).
        assert!(scanout_color_transform(pq(Some(800)), true, 203., 800.)
            .unwrap()
            .is_identity());
        // Display-referred (Windows-BT.2100) content is exempt.
        assert!(scanout_color_transform(pq(None), true, 203., 800.)
            .unwrap()
            .is_identity());

        // On SDR outputs the peak is the reference white, so regular PQ content is always
        // tone mapped (and composited).
        assert_eq!(
            scanout_color_transform(pq(Some(10_000)), false, 203., 203.),
            None
        );

        // SDR and extended-linear content is never tone mapped.
        assert!(
            scanout_color_transform(ContentColor::Sdr { gamut: bt2020 }, true, 203., 800.)
                .is_some()
        );
        assert!(scanout_color_transform(
            ContentColor::Linear {
                gamut: bt2020,
                max_lum: 1_000,
                ref_lum: 203,
            },
            true,
            203.,
            800.,
        )
        .is_some());
    }

    #[test]
    fn tonemap_curve_hits_the_output_range() {
        // f(l) = l * (1 + l*v) / (1 + l) with v solved for f(input_range) = output_range.
        let f = |l: f64, v: f64| l * (1. + l * v) / (1. + l);

        for (max_in, ref_lum, max_out) in [
            (10_000., 203., 203.),  // HDR -> SDR
            (10_000., 203., 800.),  // PQ ceiling -> 800-nit sink
            (4_000., 203., 1_000.), // mastered content -> capable sink
            (1_000., 100., 400.),   // dim reference white
        ] {
            let input_range = max_in / ref_lum;
            let output_range = max_out / ref_lum;
            let v = tonemap_curve_v(input_range, output_range);

            // The peak maps exactly onto the output peak.
            assert!((f(input_range, v) - output_range).abs() < 1e-9);
            // Black stays black and the curve never brightens.
            assert_eq!(f(0., v), 0.);
            for i in 1..=100 {
                let l = input_range * f64::from(i) / 100.;
                assert!(f(l, v) <= l * (1. + 1e-9), "f({l}) = {} > {l}", f(l, v));
            }
            // Reference white dims by at most half.
            assert!(f(1., v) >= 0.5 - 1e-9);
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

    #[test]
    fn srgb_to_pq_encoder_matches_exact_encode() {
        // The exact per-pixel encode the LUTs replace.
        fn reference(px: [u8; 4], scale: f32) -> [u8; 4] {
            let color = Color32F::new(
                f32::from(px[2]) / 255.,
                f32::from(px[1]) / 255.,
                f32::from(px[0]) / 255.,
                f32::from(px[3]) / 255.,
            );
            let color = srgb_to_pq(color, scale);
            [
                (color.b() * 255.).round().clamp(0., 255.) as u8,
                (color.g() * 255.).round().clamp(0., 255.) as u8,
                (color.r() * 255.).round().clamp(0., 255.) as u8,
                px[3],
            ]
        }

        for scale in [(203. / 10000.) as f32, (600. / 10000.) as f32, 1.] {
            let encoder = SrgbToPqEncoder::new(scale);

            // A sweep of channel values and alphas, including invalid premultiplied pixels
            // (channel > alpha) and a stride larger than the row.
            let alphas = [0u8, 1, 3, 17, 64, 128, 200, 254, 255];
            let values = (0u16..=255).step_by(5).map(|v| v as u8);
            let mut pixels = Vec::new();
            for a in alphas {
                for v in values.clone() {
                    pixels.push([v, v.wrapping_mul(3), v / 2, a]);
                    pixels.push([v, 255 - v, v.wrapping_add(97), a]);
                }
            }

            let width = 16;
            let stride = width * 4 + 12;
            let height = pixels.len().div_ceil(width);
            pixels.resize(height * width, [0; 4]);
            let mut buf = vec![0u8; height * stride];
            for (i, px) in pixels.iter().enumerate() {
                let off = (i / width) * stride + (i % width) * 4;
                buf[off..off + 4].copy_from_slice(px);
            }

            let mut encoded = buf.clone();
            encoder.apply(&mut encoded, stride as u32, (width as u32, height as u32));

            for (i, px) in pixels.iter().enumerate() {
                let off = (i / width) * stride + (i % width) * 4;
                let got: [u8; 4] = encoded[off..off + 4].try_into().unwrap();
                let want = reference(*px, scale);
                for c in 0..4 {
                    assert!(
                        got[c].abs_diff(want[c]) <= 1,
                        "scale {scale}: pixel {px:?} channel {c}: got {}, want {}",
                        got[c],
                        want[c],
                    );
                }
            }

            // Padding between rows is untouched.
            for row in 0..height {
                let pad = &encoded[row * stride + width * 4..(row + 1) * stride];
                assert!(pad.iter().all(|&b| b == 0));
            }
        }
    }
}
