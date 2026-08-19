//! Harness for tests that exercise the shaders on a real renderer.
//!
//! These tests instantiate the same renderers, shader programs and render elements that the
//! compositor uses, and actually draw them into an offscreen buffer. Compiling a shader only
//! proves that its source is valid; drawing it proves that every uniform and texture the
//! program needs is bound, because both renderers reject (GLES) or mis-render (Vulkan) draws
//! whose state was not set up. See [`super::shader_runtime`] for the tests themselves.
//!
//! Everything here works on any host: the GLES renderer is created on a surfaceless EGL
//! display, and the Vulkan renderer takes the first device that can be initialized. Set
//! `LIBGL_ALWAYS_SOFTWARE=1` to force llvmpipe, or point `VK_ICD_FILENAMES` /
//! `VK_DRIVER_FILES` at `lvp_icd.*.json` for lavapipe; hardware and software drivers are
//! equally valid here since nothing checks pixel output.
//!
//! When no renderer can be created at all (no GPU, no software driver, no EGL), the tests
//! skip themselves with a message. Set `NIRI_TEST_REQUIRE_GPU=1` to turn those skips into
//! failures, which is what CI should do once a driver is known to be installed.

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context as _;
use smithay::backend::allocator::Fourcc;
use smithay::backend::egl::native::EGLSurfacelessDisplay;
use smithay::backend::egl::{EGLContext, EGLDisplay};
use smithay::backend::renderer::damage::OutputDamageTracker;
use smithay::backend::renderer::element::RenderElement;
use smithay::backend::renderer::gles::{ffi, GlesRenderer};
use smithay::backend::renderer::vulkan::VulkanRenderer;
use smithay::backend::renderer::{Color32F, Offscreen};
use smithay::backend::vulkan::version::Version;
use smithay::backend::vulkan::{Instance, PhysicalDevice};
use smithay::utils::{Physical, Scale, Size, Transform};

use crate::render_helpers::renderer::{HasOffscreen, NiriRenderer};
use crate::render_helpers::shader_element::draw_audit;
pub use crate::render_helpers::shader_element::draw_audit::DrawRecord;
use crate::render_helpers::shaders::ProgramType;
use crate::render_helpers::{blend, resources, shaders};

/// Sends niri's warnings to stderr, once per test process.
///
/// Shader compilation and snapshot baking report their failures through the log and then fall
/// back to something that draws nothing, so without this a test failure would say only that
/// an effect is missing, not why.
pub fn init_logging() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .with_test_writer()
            .try_init();
    });
}

/// Whether a missing renderer should fail the test instead of skipping it.
fn require_gpu() -> bool {
    std::env::var_os("NIRI_TEST_REQUIRE_GPU").is_some_and(|v| !v.is_empty() && v != "0")
}

/// Reports that a test is skipped for the lack of a renderer, or panics under
/// `NIRI_TEST_REQUIRE_GPU`.
pub fn skip(what: &str, err: &anyhow::Error) {
    if require_gpu() {
        panic!("NIRI_TEST_REQUIRE_GPU is set, but {what} is unavailable: {err:?}");
    }

    // Only print once per process; every test in the file would repeat it otherwise.
    static PRINTED: AtomicBool = AtomicBool::new(false);
    if !PRINTED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "skipping GPU tests: {what} is unavailable: {err}\n\
             (install a software driver, or set NIRI_TEST_REQUIRE_GPU=1 to make this a failure)"
        );
    }
}

/// Creates a GLES renderer on a surfaceless EGL display, set up like the backends do.
///
/// Returns `None` (after reporting a skip) when no EGL implementation can give us a context,
/// which is the case on hosts without any GPU or software rasterizer.
pub fn gles_renderer() -> Option<GlesRenderer> {
    match try_gles_renderer() {
        Ok(renderer) => Some(renderer),
        Err(err) => {
            skip("the GLES renderer", &err);
            None
        }
    }
}

fn try_gles_renderer() -> anyhow::Result<GlesRenderer> {
    init_logging();

    let mut renderer = unsafe {
        let display =
            EGLDisplay::new(EGLSurfacelessDisplay).context("error creating EGL display")?;
        let context = EGLContext::new(&display).context("error creating EGL context")?;
        GlesRenderer::new(context).context("error creating renderer")?
    };

    // Same initialization as in the winit, headless and TTY backends.
    resources::init(&mut renderer);
    shaders::init(&mut renderer);
    blend::FrameBlendState::init(&mut renderer);

    Ok(renderer)
}

/// Creates a Vulkan renderer on the first device that can be initialized, set up like the TTY
/// backend does.
pub fn vulkan_renderer() -> Option<VulkanRenderer> {
    match try_vulkan_renderer() {
        Ok(renderer) => Some(renderer),
        Err(err) => {
            skip("the Vulkan renderer", &err);
            None
        }
    }
}

fn try_vulkan_renderer() -> anyhow::Result<VulkanRenderer> {
    init_logging();

    let instance =
        Instance::new(Version::VERSION_1_3, None).context("error creating Vulkan instance")?;
    let devices = PhysicalDevice::enumerate(&instance)
        .context("error enumerating Vulkan physical devices")?;

    let mut last_err = None;
    for phd in devices {
        match VulkanRenderer::new(&phd) {
            Ok(mut renderer) => {
                shaders::init_vulkan(&mut renderer);
                return Ok(renderer);
            }
            Err(err) => last_err = Some(err),
        }
    }

    match last_err {
        Some(err) => Err(anyhow::Error::new(err).context("error creating the Vulkan renderer")),
        None => anyhow::bail!("no Vulkan device found"),
    }
}

/// Renders `elements` into a fresh offscreen buffer of `size`, like the compositor renders an
/// output.
///
/// Any error from a shader draw (an unknown or mistyped uniform, a missing texture, a failed
/// pipeline) propagates out of here, and for GLES the OpenGL error state is checked as well:
/// GLES reports invalid draw state asynchronously rather than through the return value.
pub fn render_offscreen<R: NiriRenderer>(
    renderer: &mut R,
    size: Size<i32, Physical>,
    scale: f64,
    elements: &[impl RenderElement<R>],
) -> anyhow::Result<()> {
    let buffer_size = size.to_logical(1).to_buffer(1, Transform::Normal);
    let mut texture = Offscreen::<<R as HasOffscreen>::Offscreen>::create_buffer(
        renderer,
        Fourcc::Abgr8888,
        buffer_size,
    )
    .context("error creating the offscreen buffer")?;

    let mut damage = OutputDamageTracker::new(size, Scale::from(scale), Transform::Normal);

    // Anything the setup left behind is not this render's fault.
    clear_gl_errors(renderer)?;

    let res = {
        let mut target = renderer
            .bind(&mut texture)
            .context("error binding the offscreen buffer")?;
        damage
            .render_output(renderer, &mut target, 0, elements, Color32F::TRANSPARENT)
            .context("error rendering")?
    };

    // Make sure the GPU actually ran the draws before the test declares success.
    res.sync.wait().context("error waiting for the render")?;

    check_gl_error(renderer)?;

    Ok(())
}

/// Empties the OpenGL error queue of a GLES renderer.
fn clear_gl_errors<R: NiriRenderer>(renderer: &mut R) -> anyhow::Result<()> {
    let Some(renderer) = renderer.as_gles_renderer() else {
        return Ok(());
    };

    renderer
        .with_context(|gl| unsafe { while gl.GetError() != ffi::NO_ERROR {} })
        .context("error making the GL context current")
}

/// Fails if the GLES renderer (if this is one) left an OpenGL error behind.
///
/// GL reports bad draw state through this queue rather than through the return value of the
/// call that caused it, so a draw can "succeed" and still have been rejected.
pub fn check_gl_error<R: NiriRenderer>(renderer: &mut R) -> anyhow::Result<()> {
    let Some(renderer) = renderer.as_gles_renderer() else {
        return Ok(());
    };

    let errors = renderer
        .with_context(|gl| {
            let mut errors = Vec::new();
            loop {
                let err = unsafe { gl.GetError() };
                if err == ffi::NO_ERROR {
                    break errors;
                }
                errors.push(format!("{err:#x}"));
            }
        })
        .context("error making the GL context current")?;

    anyhow::ensure!(
        errors.is_empty(),
        "OpenGL errors after rendering: {}",
        errors.join(", "),
    );

    Ok(())
}

/// Uniforms of the vertex stage and of the renderer's own texture interface, which
/// [`ShaderRenderElement`](crate::render_helpers::shader_element::ShaderRenderElement) binds
/// itself rather than through the per-draw uniform list.
pub const BUILT_IN_UNIFORMS: &[&str] = &[
    "matrix",
    "tex_matrix",
    "niri_size",
    "niri_scale",
    "niri_alpha",
    "niri_tint",
];

/// Returns the names of the uniforms that a linked GL program actually uses.
///
/// Uniforms that the GLSL compiler optimized out are not active, so this is exactly the set
/// of state variables the shader reads at runtime — every one of them must be bound by
/// whoever draws with the program.
pub fn active_uniforms(renderer: &mut GlesRenderer, program: u32) -> anyhow::Result<Vec<String>> {
    renderer
        .with_context(|gl| unsafe {
            let mut count = 0;
            gl.GetProgramiv(program, ffi::ACTIVE_UNIFORMS, &mut count);
            let mut max_len = 0;
            gl.GetProgramiv(program, ffi::ACTIVE_UNIFORM_MAX_LENGTH, &mut max_len);

            let mut names = Vec::new();
            for index in 0..count.max(0) as u32 {
                let mut buf = vec![0u8; max_len.max(1) as usize];
                let mut len = 0;
                let mut size = 0;
                let mut type_ = 0;
                gl.GetActiveUniform(
                    program,
                    index,
                    buf.len() as i32,
                    &mut len,
                    &mut size,
                    &mut type_,
                    buf.as_mut_ptr().cast(),
                );

                let name = String::from_utf8_lossy(&buf[..len.max(0) as usize]).into_owned();
                // Array uniforms are reported as "name[0]".
                let name = name.split('[').next().unwrap_or(&name).to_owned();
                names.push(name);
            }
            names
        })
        .context("error making the GL context current")
}

/// Checks that every uniform the drawn programs actually use was bound by the draw.
///
/// This is the check that a plain draw cannot make: GLES rejects a uniform that the program
/// never declared, but it silently ignores a declared uniform that nobody set, leaving the
/// shader to read whatever the program object happened to hold. Uniforms added to a shader
/// (or to the shared blend stage) but never wired up in the element that draws it show up
/// here, and nowhere else.
pub fn check_uniform_coverage(
    renderer: &mut GlesRenderer,
    records: &[DrawRecord],
) -> anyhow::Result<()> {
    for record in records {
        let Some(gl_program) = record.gl_program else {
            // A Vulkan draw; nothing to introspect, its uniform block is filled by offset.
            continue;
        };

        let active = active_uniforms(renderer, gl_program)?;
        let missing: Vec<_> = active
            .iter()
            .filter(|name| {
                !record.uniforms.iter().any(|bound| bound == *name)
                    && !record.textures.iter().any(|bound| bound == *name)
                    && !BUILT_IN_UNIFORMS.contains(&name.as_str())
            })
            .collect();

        anyhow::ensure!(
            missing.is_empty(),
            "the {:?} shader uses uniforms that the draw did not bind: {missing:?}\n\
             (bound: {:?}, textures: {:?})",
            record.program,
            record.uniforms,
            record.textures,
        );
    }

    Ok(())
}

/// What a render pass drew.
#[derive(Debug, Default)]
pub struct Audit {
    /// Draws of niri's own shader programs, with the uniforms each one bound.
    pub draws: Vec<DrawRecord>,
    /// Names of the custom texture programs that were drawn with.
    pub texture_programs: Vec<&'static str>,
}

impl Audit {
    /// Fails unless each of `expected` was drawn.
    ///
    /// Scenes are assembled by the layout from the config, so an effect can quietly stop
    /// being rendered; without this a test would keep passing while covering nothing.
    #[track_caller]
    pub fn assert_drew(&self, expected: &[ProgramType]) {
        for program in expected {
            assert!(
                self.draws.iter().any(|draw| draw.program == *program),
                "the {program:?} shader was never drawn; drawn: {:?}, texture programs: {:?}",
                self.draws.iter().map(|d| d.program).collect::<Vec<_>>(),
                self.texture_programs,
            );
        }
    }

    /// Fails unless each of the named texture programs was drawn with.
    #[track_caller]
    pub fn assert_drew_texture(&self, expected: &[&str]) {
        for program in expected {
            assert!(
                self.texture_programs.contains(program),
                "the {program} shader was never drawn; texture programs: {:?}",
                self.texture_programs,
            );
        }
    }
}

/// [`render_offscreen`] with the shader draw audit running: returns what was drawn, after
/// checking that every draw bound everything its program uses.
pub fn render_offscreen_audited<R: NiriRenderer>(
    renderer: &mut R,
    size: Size<i32, Physical>,
    scale: f64,
    elements: &[impl RenderElement<R>],
) -> anyhow::Result<Audit> {
    draw_audit::start();
    let res = render_offscreen(renderer, size, scale, elements);
    let audit = Audit {
        draws: draw_audit::take(),
        texture_programs: draw_audit::take_texture_programs(),
    };
    res?;

    if let Some(renderer) = renderer.as_gles_renderer() {
        check_uniform_coverage(renderer, &audit.draws)?;
    }

    Ok(audit)
}
