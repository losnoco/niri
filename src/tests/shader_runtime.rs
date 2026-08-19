//! Runtime tests for the shaders.
//!
//! Compiling a shader proves that its source is valid. It does not prove that the code
//! drawing with it sets up everything the shader reads: a uniform that was renamed, retyped
//! or forgotten is only discovered when a draw actually happens, on a renderer, on a device.
//! These tests do that draw, using the same shader programs and render elements the
//! compositor uses, and they check that every uniform the program uses was bound.
//!
//! They need a renderer, but not a particular one: a software rasterizer (llvmpipe,
//! lavapipe) works as well as a GPU, so they run on any host and in CI. Without any usable
//! driver they skip themselves; see [`super::gpu`] for the harness and for
//! `NIRI_TEST_REQUIRE_GPU`, which turns skips into failures.

use std::time::Duration;

use niri_config::{Color, Config, CornerRadius, GradientInterpolation};
use niri_ipc::SizeChange;
use smithay::backend::renderer::element::{Element as _, Kind};
use smithay::utils::{Physical, Point, Rectangle, Scale, Size};
use wayland_client::protocol::wl_surface::WlSurface;

use super::client::ClientId;
use super::{gpu, Fixture};
use crate::backend::tty_renderer::TtyOffscreen;
use crate::render_helpers::background_effect::RenderParams;
use crate::render_helpers::blur::BlurOptions;
use crate::render_helpers::border::BorderRenderElement;
use crate::render_helpers::framebuffer_effect::{FramebufferEffect, FramebufferEffectElement};
use crate::render_helpers::offscreen::OffscreenBuffer;
use crate::render_helpers::renderer::NiriCaptureRenderer;
use crate::render_helpers::resize::ResizeRenderElement;
use crate::render_helpers::shaders::ProgramType;
use crate::render_helpers::shadow::ShadowRenderElement;
use crate::render_helpers::solid_color::{SolidColorBuffer, SolidColorRenderElement};
use crate::render_helpers::{blend, shaders, RenderCtx, RenderTarget};

/// Reference and peak luminance of an HDR frame, in cd/m².
///
/// Every shader carries the shared blend stage, whose uniforms only do something in HDR
/// frames, so the whole test suite runs twice: once in SDR, once in this blend space.
const HDR_BLEND: Option<(f64, f64)> = Some((203., 1000.));

// Custom animation shaders, in the shape users write them (the wiki examples are prose with
// placeholders, so they can't be compiled directly). Each one reads every uniform and texture
// its interface offers, so that the coverage check has something to verify: a uniform the
// GLSL never mentions is optimized out of the linked program and proves nothing.

// An expanding circle, like the wiki example, with a per-window jitter.
const CUSTOM_OPEN_SHADER: &str = "\
vec4 open_color(vec3 coords_geo, vec3 size_geo) {
    vec3 coords_tex = niri_geo_to_tex * coords_geo;
    vec4 color = texture2D(niri_tex, coords_tex.st);

    vec2 coords = (coords_geo.xy - vec2(0.5)) * size_geo.xy * 2.0 / length(size_geo.xy);
    float radius = niri_clamped_progress * (0.9 + 0.2 * fract(niri_random_seed));
    if (radius * radius <= dot(coords, coords))
        color = vec4(0.0);

    // Fade in on the unclamped progress, so that uniform is read as well.
    return color * clamp(niri_progress, 0.0, 1.0);
}
";

// The window slides down by a per-window amount while it fades out.
const CUSTOM_CLOSE_SHADER: &str = "\
vec4 close_color(vec3 coords_geo, vec3 size_geo) {
    float shift = niri_clamped_progress * (0.5 + fract(niri_random_seed));
    coords_geo = vec3(coords_geo.xy - vec2(0.0, shift * 20.0 / size_geo.y), 1.0);

    vec3 coords_tex = niri_geo_to_tex * coords_geo;
    vec4 color = texture2D(niri_tex, coords_tex.st);

    return color * clamp(1.0 - niri_progress, 0.0, 1.0);
}
";

// A crossfade between the previous and the next window contents.

const CUSTOM_RESIZE_SHADER: &str = "\
vec4 resize_color(vec3 coords_curr_geo, vec3 size_curr_geo) {
    vec3 coords_next = niri_geo_to_tex_next * niri_curr_geo_to_next_geo * coords_curr_geo;
    vec3 coords_prev = niri_geo_to_tex_prev * niri_curr_geo_to_prev_geo * coords_curr_geo;

    vec4 color = texture2D(niri_tex_next, coords_next.st);
    vec4 color_prev = texture2D(niri_tex_prev, coords_prev.st);
    color = mix(color_prev, color, niri_clamped_progress);

    return color * clamp(niri_progress, 0.0, 1.0);
}
";

// =============================================================================
// Individual shader elements.
// =============================================================================

#[test]
fn border_draws_gles() {
    let Some(mut renderer) = gpu::gles_renderer() else {
        return;
    };

    for blend in [None, HDR_BLEND] {
        blend::set_frame_blend(&mut renderer, blend);

        gpu::render_offscreen_audited(&mut renderer, size(), 1., &[border_element()])
            .unwrap()
            .assert_drew(&[ProgramType::Border]);
    }
}

#[test]
fn border_draws_vulkan() {
    let Some(mut renderer) = gpu::vulkan_renderer() else {
        return;
    };

    for blend in [None, HDR_BLEND] {
        blend::set_frame_blend_vulkan(&mut renderer, blend);

        gpu::render_offscreen_audited(&mut renderer, size(), 1., &[border_element()])
            .unwrap()
            .assert_drew(&[ProgramType::Border]);
    }
}

#[test]
fn shadow_draws_gles() {
    let Some(mut renderer) = gpu::gles_renderer() else {
        return;
    };

    for blend in [None, HDR_BLEND] {
        blend::set_frame_blend(&mut renderer, blend);

        gpu::render_offscreen_audited(&mut renderer, size(), 1., &[shadow_element()])
            .unwrap()
            .assert_drew(&[ProgramType::Shadow]);
    }
}

#[test]
fn shadow_draws_vulkan() {
    let Some(mut renderer) = gpu::vulkan_renderer() else {
        return;
    };

    for blend in [None, HDR_BLEND] {
        blend::set_frame_blend_vulkan(&mut renderer, blend);

        gpu::render_offscreen_audited(&mut renderer, size(), 1., &[shadow_element()])
            .unwrap()
            .assert_drew(&[ProgramType::Shadow]);
    }
}

/// Draws the resize animation shader, both the built-in one and a user's custom one.
///
/// The custom shader goes through the same prelude and epilogue that a user's
/// `custom-shader` does, so this covers the whole interface niri promises them.
#[test]
fn resize_draws_gles() {
    let Some(mut renderer) = gpu::gles_renderer() else {
        return;
    };

    for custom in [None, Some(CUSTOM_RESIZE_SHADER)] {
        shaders::set_custom_resize_program(&mut renderer, custom);

        for blend in [None, HDR_BLEND] {
            blend::set_frame_blend(&mut renderer, blend);

            let (prev, next) = (offscreen(&mut renderer), offscreen(&mut renderer));
            let elem = resize_element(&prev, &next);
            gpu::render_offscreen_audited(&mut renderer, size(), 1., &[elem])
                .unwrap()
                .assert_drew(&[ProgramType::Resize]);
        }
    }

    shaders::set_custom_resize_program(&mut renderer, None);
}

#[test]
fn resize_draws_vulkan() {
    let Some(mut renderer) = gpu::vulkan_renderer() else {
        return;
    };

    for custom in [None, Some(CUSTOM_RESIZE_SHADER)] {
        shaders::set_custom_resize_program(&mut renderer, custom);

        for blend in [None, HDR_BLEND] {
            blend::set_frame_blend_vulkan(&mut renderer, blend);

            let (prev, next) = (offscreen(&mut renderer), offscreen(&mut renderer));
            let elem = resize_element(&prev, &next);
            gpu::render_offscreen_audited(&mut renderer, size(), 1., &[elem])
                .unwrap()
                .assert_drew(&[ProgramType::Resize]);
        }
    }

    shaders::set_custom_resize_program(&mut renderer, None);
}

/// Draws the framebuffer effect: the blur and postprocess programs that sit behind
/// semi-transparent windows and layer surfaces.
#[test]
fn framebuffer_effect_draws_gles() {
    let Some(mut renderer) = gpu::gles_renderer() else {
        return;
    };

    for blend in [None, HDR_BLEND] {
        blend::set_frame_blend(&mut renderer, blend);

        gpu::render_offscreen_audited(&mut renderer, size(), 1., &[framebuffer_effect_element()])
            .unwrap()
            .assert_drew_texture(&["blur", "postprocess_and_clip"]);
    }
}

#[test]
fn framebuffer_effect_draws_vulkan() {
    let Some(mut renderer) = gpu::vulkan_renderer() else {
        return;
    };

    for blend in [None, HDR_BLEND] {
        blend::set_frame_blend_vulkan(&mut renderer, blend);

        gpu::render_offscreen_audited(&mut renderer, size(), 1., &[framebuffer_effect_element()])
            .unwrap()
            .assert_drew_texture(&["blur", "postprocess_and_clip"]);
    }
}

// =============================================================================
// Whole scenes, assembled by the compositor.
// =============================================================================

/// Renders an output with every shader-backed decoration enabled.
///
/// Unlike the tests above, nothing here constructs render elements by hand: the layout does,
/// from the config, exactly as it does on a real output. This is what catches an element that
/// the compositor builds differently from how its shader expects.
#[test]
fn scene_with_effects_draws_gles() {
    let Some(mut f) = set_up(effects_config()) else {
        return;
    };
    add_window(&mut f);
    settle(&mut f);

    let mut drawn = Vec::new();
    for blend in [None, HDR_BLEND] {
        let audit = render_scene(&mut f, blend);
        audit.assert_drew(&[ProgramType::Border, ProgramType::Shadow]);
        drawn.extend(audit.texture_programs);
    }

    // The window's own texture goes through the rounded-corner clip, and the blur behind it
    // through the blur and postprocess programs. Effects that cache their result only render
    // on the first pass, so the passes are checked together.
    assert!(drawn.contains(&"clipped_surface"), "drawn: {drawn:?}");
    assert!(drawn.contains(&"blur"), "drawn: {drawn:?}");
    assert!(drawn.contains(&"postprocess_and_clip"), "drawn: {drawn:?}");
}

/// Renders the window opening animation, with and without a custom shader.
#[test]
fn scene_window_open_draws_gles() {
    for custom in [None, Some(CUSTOM_OPEN_SHADER)] {
        let mut config = effects_config();
        config.animations.window_open.custom_shader = custom.map(str::to_owned);

        let Some(mut f) = set_up(config) else {
            return;
        };
        add_window(&mut f);

        // Halfway into the 1s opening animation.
        set_time(&mut f, Duration::from_millis(500));

        for blend in [None, HDR_BLEND] {
            let audit = render_scene(&mut f, blend);
            if custom.is_some() {
                audit.assert_drew(&[ProgramType::Open]);
            }
        }
    }
}

/// Renders the window closing animation, with and without a custom shader.
#[test]
fn scene_window_close_draws_gles() {
    for custom in [None, Some(CUSTOM_CLOSE_SHADER)] {
        let mut config = effects_config();
        config.animations.window_close.custom_shader = custom.map(str::to_owned);

        let Some(mut f) = set_up(config) else {
            return;
        };
        let (id, surface) = add_window(&mut f);
        settle(&mut f);

        // Unmapping snapshots the window into a closing animation.
        let window = f.client(id).window(&surface);
        window.attach_null();
        window.commit();
        f.double_roundtrip(id);

        set_time(&mut f, Duration::from_millis(500));

        for blend in [None, HDR_BLEND] {
            let audit = render_scene(&mut f, blend);
            if custom.is_some() {
                audit.assert_drew(&[ProgramType::Close]);
            }
        }
    }
}

/// Renders the window resize animation, with and without a custom shader.
#[test]
fn scene_window_resize_draws_gles() {
    for custom in [None, Some(CUSTOM_RESIZE_SHADER)] {
        let mut config = effects_config();
        config.animations.window_resize.custom_shader = custom.map(str::to_owned);

        let Some(mut f) = set_up(config) else {
            return;
        };
        let (id, surface) = add_window(&mut f);
        settle(&mut f);

        f.niri()
            .layout
            .set_window_height(None, SizeChange::AdjustFixed(-100));
        f.double_roundtrip(id);

        ack_configured_size(&mut f, id, &surface);
        f.double_roundtrip(id);

        set_time(&mut f, Duration::from_millis(500));

        for blend in [None, HDR_BLEND] {
            let audit = render_scene(&mut f, blend);
            audit.assert_drew(&[ProgramType::Resize]);
        }
    }
}

/// Renders the overview, which draws the workspace backdrop and shadows.
#[test]
fn scene_overview_draws_gles() {
    let Some(mut f) = set_up(effects_config()) else {
        return;
    };
    add_window(&mut f);
    settle(&mut f);

    f.niri().layout.toggle_overview();
    f.niri_complete_animations();

    let mut drawn = Vec::new();
    for blend in [None, HDR_BLEND] {
        let audit = render_scene(&mut f, blend);
        audit.assert_drew(&[ProgramType::Border, ProgramType::Shadow]);
        drawn.extend(audit.texture_programs);
    }

    assert!(drawn.contains(&"clipped_surface"), "drawn: {drawn:?}");
    assert!(drawn.contains(&"postprocess_and_clip"), "drawn: {drawn:?}");
}

/// Renders a window whose contents are HDR (PQ, BT.2020), like a video player's.
///
/// HDR content is the one case where the texture shaders do more than pass through: into an
/// HDR frame it is tone mapped down to the output's peak, and into an SDR frame (a
/// screenshot, a screencast) it is converted back to SDR.
#[test]
fn scene_hdr_content_draws_gles() {
    use niri_config::output::Hdr;
    use smithay::reexports::wayland_protocols::wp::color_management::v1::client::wp_color_manager_v1::{
        Primaries, RenderIntent, TransferFunction,
    };

    // No corner radius here: a clipped surface runs the blend stage inside its own shader,
    // while an unclipped one goes through the renderer's texture program, which is what this
    // test is after.
    let mut config = Config::parse_mem(
        r"
        layout {
            border {
                on
                width 4
            }
        }
        ",
    )
    .unwrap();
    config.outputs.0.push(niri_config::Output {
        name: "headless-1".to_owned(),
        hdr: Some(Hdr::default()),
        ..Default::default()
    });

    let Some(mut f) = set_up(config) else {
        return;
    };
    let (id, surface) = add_window(&mut f);

    f.client(id).create_and_attach_hdr_description(
        &surface,
        TransferFunction::St2084Pq,
        Primaries::Bt2020,
        RenderIntent::Perceptual,
    );
    f.client(id).window(&surface).commit();
    f.double_roundtrip(id);
    settle(&mut f);

    // An HDR frame whose peak is below the content's: the content is tone mapped.
    render_scene(&mut f, Some((203., 400.))).assert_drew_texture(&["texture_hdr"]);

    // An SDR frame: the same content is converted back to SDR.
    render_scene(&mut f, None).assert_drew_texture(&["texture_hdr_to_sdr"]);
}

/// Renders the recent-windows (MRU) overlay, which fades window previews out with the
/// gradient-fade shader.
#[test]
fn scene_recent_windows_draws_gles() {
    use crate::ui::mru::WindowMru;

    let mut config = effects_config();
    // The overlay only renders once its open delay has passed.
    config.recent_windows.open_delay_ms = 0;

    let Some(mut f) = set_up(config) else {
        return;
    };
    for _ in 0..2 {
        let (id, surface) = add_window(&mut f);
        // The overlay fades out long titles with the gradient shader, so give it one.
        f.client(id)
            .window(&surface)
            .xdg_toplevel
            .set_title("a window with a fairly long title to fade out".to_owned());
        f.client(id).window(&surface).commit();
        f.double_roundtrip(id);
    }
    settle(&mut f);

    // Same as the switch-focus bind, without going through the input path.
    {
        let niri = f.niri();
        let wmru = WindowMru::new(niri);
        assert!(!wmru.is_empty(), "no windows for the MRU overlay");
        let clock = niri.clock.clone();
        let output = niri.layout.active_output().unwrap().clone();
        niri.window_mru_ui.open(clock, wmru, output);
    }
    settle(&mut f);

    let mut drawn = Vec::new();
    for blend in [None, HDR_BLEND] {
        drawn.extend(render_scene(&mut f, blend).texture_programs);
    }

    assert!(drawn.contains(&"gradient_fade"), "drawn: {drawn:?}");
}

/// Renders a scene with a layer-shell background and a window blurring it through the xray
/// path, which samples the backdrop once instead of the whole framebuffer.
#[test]
fn scene_xray_layer_surface_draws_gles() {
    use smithay::reexports::wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_shell_v1::Layer;
    use smithay::reexports::wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1::Anchor;

    use super::client::LayerConfigureProps;

    let config = r##"
        layout {
            border {
                on
                width 4
            }
        }

        window-rule {
            geometry-corner-radius 12
            clip-to-geometry true
            opacity 0.8
            background-effect {
                xray true
                blur true
            }
        }
    "##;
    let Some(mut f) = set_up(Config::parse_mem(config).unwrap()) else {
        return;
    };

    // A wallpaper for the xray blur to sample.
    let id = f.add_client();
    let layer = f
        .client(id)
        .create_layer(None, Layer::Background, "wallpaper");
    let surface = layer.surface.clone();
    layer.set_configure_props(LayerConfigureProps {
        anchor: Some(Anchor::Left | Anchor::Right | Anchor::Top | Anchor::Bottom),
        ..Default::default()
    });
    layer.commit();
    f.roundtrip(id);

    let layer = f.client(id).layer(&surface);
    let (w, h) = layer.configures_received.last().unwrap().1.size;
    layer.attach_new_shm_buffer(w as u16, h as u16);
    layer.set_size(w as u16, h as u16);
    layer.ack_last_and_commit();
    f.double_roundtrip(id);

    add_window(&mut f);
    settle(&mut f);

    let mut drawn = Vec::new();
    for blend in [None, HDR_BLEND] {
        let audit = render_scene(&mut f, blend);
        audit.assert_drew(&[ProgramType::Border]);
        drawn.extend(audit.texture_programs);
    }

    assert!(drawn.contains(&"blur"), "drawn: {drawn:?}");
    assert!(drawn.contains(&"postprocess_and_clip"), "drawn: {drawn:?}");
    assert!(drawn.contains(&"clipped_surface"), "drawn: {drawn:?}");
}

// =============================================================================
// Helpers.
// =============================================================================

fn size() -> Size<i32, Physical> {
    Size::from((256, 256))
}

fn border_element() -> BorderRenderElement {
    BorderRenderElement::new(
        Size::from((200., 200.)),
        Rectangle::new(Point::from((0., 0.)), Size::from((200., 200.))),
        GradientInterpolation::default(),
        Color::from_rgba8_unpremul(255, 0, 0, 255),
        Color::from_rgba8_unpremul(0, 0, 255, 255),
        0.5,
        Rectangle::new(Point::from((10., 10.)), Size::from((180., 180.))),
        4.,
        CornerRadius::from(8.),
        1.,
        1.,
    )
}

fn shadow_element() -> ShadowRenderElement {
    ShadowRenderElement::new(
        Size::from((200., 200.)),
        Rectangle::new(Point::from((0., 0.)), Size::from((200., 200.))),
        Color::from_rgba8_unpremul(0, 0, 0, 128),
        8.,
        CornerRadius::from(8.),
        1.,
        Rectangle::new(Point::from((10., 10.)), Size::from((180., 180.))),
        CornerRadius::from(4.),
        1.,
    )
}

/// A window's background effect: blurs whatever is already in the framebuffer, then applies
/// the postprocess (noise, saturation) and the rounded-corner clip.
fn framebuffer_effect_element() -> FramebufferEffectElement {
    let geometry = Rectangle::new(Point::from((20., 20.)), Size::from((200., 200.)));

    FramebufferEffect::new().render(
        None,
        RenderParams {
            geometry,
            subregion: None,
            clip: Some((geometry, CornerRadius::from(12.))),
            scale: 1.,
        },
        Some(BlurOptions {
            passes: 2,
            offset: 3.,
        }),
        0.05,
        1.5,
    )
}

/// A texture with something rendered into it, to feed the animation shaders.
///
/// The buffer is returned along with the texture because it owns it: dropping it while the
/// element still refers to the texture would recreate it on the next render.
struct Offscreen {
    _buffer: OffscreenBuffer,
    texture: TtyOffscreen,
    geo: Rectangle<i32, Physical>,
}

fn offscreen<R: NiriCaptureRenderer>(renderer: &mut R) -> Offscreen
where
    R::Error: Send + Sync + 'static,
{
    let buffer = OffscreenBuffer::default();
    let color = SolidColorBuffer::new(Size::from((100., 100.)), [0.5, 0.2, 0.8, 1.]);
    let elem =
        SolidColorRenderElement::from_buffer(&color, Point::from((0., 0.)), 1., Kind::Unspecified);

    let (elem, _sync, _data) = buffer
        .render(renderer, Scale::from(1.), &[elem])
        .expect("error rendering the offscreen buffer");

    Offscreen {
        texture: elem.texture().clone(),
        geo: elem.geometry(Scale::from(1.)),
        _buffer: buffer,
    }
}

fn resize_element(prev: &Offscreen, next: &Offscreen) -> ResizeRenderElement {
    ResizeRenderElement::new(
        Rectangle::new(Point::from((0., 0.)), Size::from((100., 100.))),
        Scale::from(1.),
        (prev.texture.clone(), prev.geo),
        Size::from((100., 150.)),
        (next.texture.clone(), next.geo),
        Size::from((100., 100.)),
        0.5,
        0.5,
        CornerRadius::from(8.),
        true,
        1.,
    )
}

/// A config with every shader-backed decoration turned on.
fn effects_config() -> Config {
    let config = r##"
        layout {
            gaps 16
            focus-ring {
                width 4
                active-gradient from="#f00" to="#00f" angle=45 relative-to="workspace-view"
            }
            border {
                on
                width 4
                active-gradient from="#f0f" to="#0ff" angle=180 in="oklch longer hue"
            }
            shadow {
                on
                softness 30
                spread 5
                offset x=0 y=5
            }
            tab-indicator {
                width 4
                gap 4
                active-gradient from="#f00" to="#00f" angle=45
            }
        }

        window-rule {
            geometry-corner-radius 12
            clip-to-geometry true
            background-effect {
                blur true
                noise 0.05
                saturation 3
            }
        }

        animations {
            window-open {
                duration-ms 1000
                curve "linear"
            }
            window-close {
                duration-ms 1000
                curve "linear"
            }
            window-resize {
                duration-ms 1000
                curve "linear"
            }
        }
    "##;
    Config::parse_mem(config).unwrap()
}

/// Sets up a fixture with a renderer, the config's custom shaders, and one output.
///
/// Returns `None` (after reporting a skip) when the host has no renderer, same as
/// [`gpu::gles_renderer`].
fn set_up(config: Config) -> Option<Fixture> {
    gpu::init_logging();

    let resize = config.animations.window_resize.custom_shader.clone();
    let close = config.animations.window_close.custom_shader.clone();
    let open = config.animations.window_open.custom_shader.clone();

    let mut f = Fixture::with_config(config);
    if let Err(err) = f.niri_state().backend.headless().add_renderer() {
        gpu::skip("the headless backend's renderer", &err);
        return None;
    }

    // The headless backend doesn't do this itself, unlike the winit and TTY backends.
    let renderer = f.niri_state().backend.headless().renderer().unwrap();
    shaders::set_custom_resize_program(renderer, resize.as_deref());
    shaders::set_custom_close_program(renderer, close.as_deref());
    shaders::set_custom_open_program(renderer, open.as_deref());

    f.add_output(1, (800, 600));
    Some(f)
}

/// Maps a window with a real (textured) buffer attached, and puts the clock at zero.
///
/// The window is left in its opening animation; call [`settle`] for a static scene.
fn add_window(f: &mut Fixture) -> (ClientId, WlSurface) {
    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);

    let window = f.client(id).window(&surface);
    let (w, h) = window.configures_received.last().unwrap().1.size;
    window.attach_new_shm_buffer(w as u16, h as u16);
    ack_configured_size(f, id, &surface);
    f.double_roundtrip(id);

    set_time(f, Duration::ZERO);

    (id, surface)
}

/// Acks the last configure and commits at the size it asked for.
///
/// The size has to follow, or the compositor sees no size change and skips the resize
/// animation that some of these tests are after.
fn ack_configured_size(f: &mut Fixture, id: ClientId, surface: &WlSurface) {
    let window = f.client(id).window(surface);
    let (w, h) = window.configures_received.last().unwrap().1.size;
    window.set_size(w as u16, h as u16);
    window.ack_last_and_commit();
}

/// Runs every animation to its end, leaving a settled scene.
fn settle(f: &mut Fixture) {
    f.niri_complete_animations();
}

/// Puts the animation clock at `time`, then advances the animations to it.
///
/// Same dance as in the animation tests: the clock is adjustable and keeps its own current
/// time, so it has to be reset to zero before the wanted time can be set.
fn set_time(f: &mut Fixture, time: Duration) {
    let niri = f.niri();

    let now = niri.clock.now();
    niri.clock.set_unadjusted(now);
    let _ = niri.clock.now();
    niri.clock.set_unadjusted(Duration::ZERO);
    niri.clock.set_rate(1.0);
    let _ = niri.clock.now();

    niri.clock.set_unadjusted(time);
    let _ = niri.clock.now();

    // Freeze so that the niri loop callback doesn't replace it with the monotonic time.
    niri.clock.set_rate(0.0);

    niri.advance_animations();
}

/// Renders everything the compositor would put on the output, in the given blend space.
fn render_scene(f: &mut Fixture, blend: Option<(f64, f64)>) -> gpu::Audit {
    let output = f.niri_output(1);

    // The compositor does this at the top of every redraw; the layout checks that render
    // elements were updated at the time it is now rendering at.
    f.niri().update_render_elements(Some(&output));

    let size = output.current_mode().unwrap().size;
    let state = f.niri_state();

    let elements = {
        let renderer = state.backend.headless().renderer().unwrap();
        blend::set_frame_blend(renderer, blend);

        let ctx = RenderCtx {
            renderer,
            target: RenderTarget::Output,
            xray: None,
        };
        state.niri.render_to_vec(ctx, &output, true)
    };

    let renderer = state.backend.headless().renderer().unwrap();
    let audit = gpu::render_offscreen_audited(renderer, size, 1., &elements).unwrap();

    blend::set_frame_blend(renderer, None);

    audit
}
