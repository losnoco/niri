use std::cell::RefCell;

use glam::Mat3;
use smithay::backend::renderer::gles::{
    GlesFrame, GlesRenderer, GlesTexProgram, Uniform, UniformName, UniformType, UniformValue,
};
use smithay::backend::renderer::vulkan::{
    texture_bindings_glsl, uniform_block_glsl, CustomUniformDecl, CustomUniformKind,
    VulkanPixelProgram, VulkanRenderer,
};

use super::blend::FrameBlendState;
use super::renderer::NiriRenderer;
use super::shader_element::ShaderProgram;
use crate::render_helpers::blur::BlurProgram;

/// A custom texture shader program for either renderer.
#[derive(Debug, Clone)]
pub enum NiriTexProgram {
    Gles(GlesTexProgram),
    Vulkan(VulkanPixelProgram),
}

pub struct Shaders {
    pub texture_hdr: Option<GlesTexProgram>,
    pub texture_hdr_to_sdr: Option<GlesTexProgram>,
    pub border: Option<ShaderProgram>,
    pub shadow: Option<ShaderProgram>,
    pub clipped_surface: Option<NiriTexProgram>,
    pub postprocess_and_clip: Option<GlesTexProgram>,
    pub resize: Option<ShaderProgram>,
    pub gradient_fade: Option<GlesTexProgram>,
    pub blur: Option<BlurProgram>,
    pub custom_resize: RefCell<Option<ShaderProgram>>,
    pub custom_close: RefCell<Option<ShaderProgram>>,
    pub custom_open: RefCell<Option<ShaderProgram>>,
}

#[derive(Debug, Clone, Copy)]
pub enum ProgramType {
    Border,
    Shadow,
    Resize,
    Close,
    Open,
}

/// A program's own uniform declarations followed by the shared `niri_blend` ones.
///
/// Every program compiled here embeds (or is drawn with) the blend stage, so its uniform list
/// must always include [`FrameBlendState::uniform_names`]; missing declarations fail draws at
/// runtime, not compile time.
fn with_blend_uniform_names(specific: &[UniformName<'static>]) -> Vec<UniformName<'static>> {
    let mut names = specific.to_vec();
    names.extend(FrameBlendState::uniform_names());
    names
}

impl Shaders {
    fn compile(renderer: &mut GlesRenderer) -> Self {
        let _span = tracy_client::span!("Shaders::compile");

        let texture_hdr = renderer
            .compile_custom_texture_shader(
                concat!(include_str!("texture_hdr.frag"), include_str!("hdr.frag"),),
                &FrameBlendState::uniform_names(),
            )
            .map_err(|err| {
                warn!("error compiling HDR texture shader: {err:?}");
            })
            .ok();

        let texture_hdr_to_sdr = renderer
            .compile_custom_texture_shader(
                include_str!("texture_hdr_to_sdr.frag"),
                &FrameBlendState::uniform_names(),
            )
            .map_err(|err| {
                warn!("error compiling HDR-to-SDR texture shader: {err:?}");
            })
            .ok();

        let border = ShaderProgram::compile(
            renderer,
            concat!(
                include_str!("border.frag"),
                include_str!("rounding_alpha.frag")
            ),
            &[
                UniformName::new("colorspace", UniformType::_1f),
                UniformName::new("hue_interpolation", UniformType::_1f),
                UniformName::new("color_from", UniformType::_4f),
                UniformName::new("color_to", UniformType::_4f),
                UniformName::new("grad_offset", UniformType::_2f),
                UniformName::new("grad_width", UniformType::_1f),
                UniformName::new("grad_vec", UniformType::_2f),
                UniformName::new("input_to_geo", UniformType::Matrix3x3),
                UniformName::new("geo_size", UniformType::_2f),
                UniformName::new("outer_radius", UniformType::_4f),
                UniformName::new("border_width", UniformType::_1f),
            ],
            &[],
        )
        .map_err(|err| {
            warn!("error compiling border shader: {err:?}");
        })
        .ok();

        let shadow = ShaderProgram::compile(
            renderer,
            concat!(
                include_str!("shadow.frag"),
                include_str!("rounding_alpha.frag")
            ),
            &[
                UniformName::new("shadow_color", UniformType::_4f),
                UniformName::new("sigma", UniformType::_1f),
                UniformName::new("input_to_geo", UniformType::Matrix3x3),
                UniformName::new("geo_size", UniformType::_2f),
                UniformName::new("corner_radius", UniformType::_4f),
                UniformName::new("window_input_to_geo", UniformType::Matrix3x3),
                UniformName::new("window_geo_size", UniformType::_2f),
                UniformName::new("window_corner_radius", UniformType::_4f),
            ],
            &[],
        )
        .map_err(|err| {
            warn!("error compiling shadow shader: {err:?}");
        })
        .ok();

        let clipped_surface = renderer
            .compile_custom_texture_shader(
                concat!(
                    include_str!("clipped_surface.frag"),
                    include_str!("rounding_alpha.frag"),
                    include_str!("hdr.frag"),
                    "\nvec4 postprocess(vec4 color) { return color; }",
                ),
                &with_blend_uniform_names(&[
                    UniformName::new("niri_scale", UniformType::_1f),
                    UniformName::new("geo_size", UniformType::_2f),
                    UniformName::new("corner_radius", UniformType::_4f),
                    UniformName::new("input_to_geo", UniformType::Matrix3x3),
                ]),
            )
            .map_err(|err| {
                warn!("error compiling clipped surface shader: {err:?}");
            })
            .ok()
            .map(NiriTexProgram::Gles);

        let postprocess_and_clip = renderer
            .compile_custom_texture_shader(
                concat!(
                    include_str!("clipped_surface.frag"),
                    include_str!("rounding_alpha.frag"),
                    include_str!("postprocess.frag"),
                    include_str!("hdr.frag"),
                ),
                &with_blend_uniform_names(&[
                    UniformName::new("niri_scale", UniformType::_1f),
                    UniformName::new("geo_size", UniformType::_2f),
                    UniformName::new("corner_radius", UniformType::_4f),
                    UniformName::new("input_to_geo", UniformType::Matrix3x3),
                    UniformName::new("noise", UniformType::_1f),
                    UniformName::new("saturation", UniformType::_1f),
                    UniformName::new("bg_color", UniformType::_4f),
                ]),
            )
            .map_err(|err| {
                warn!("error compiling postprocess_and_clip shader: {err:?}");
            })
            .ok();

        let resize = compile_resize_program(renderer, include_str!("resize.frag"))
            .map_err(|err| {
                warn!("error compiling resize shader: {err:?}");
            })
            .ok();

        let gradient_fade = renderer
            .compile_custom_texture_shader(
                concat!(include_str!("gradient_fade.frag"), include_str!("hdr.frag")),
                &with_blend_uniform_names(&[UniformName::new("cutoff", UniformType::_2f)]),
            )
            .map_err(|err| {
                warn!("error compiling gradient fade shader: {err:?}");
            })
            .ok();

        let blur = BlurProgram::compile(renderer)
            .map_err(|err| {
                warn!("error compiling blur shaders: {err:?}");
            })
            .ok();

        Self {
            texture_hdr,
            texture_hdr_to_sdr,
            border,
            shadow,
            clipped_surface,
            postprocess_and_clip,
            resize,
            gradient_fade,
            blur,
            custom_resize: RefCell::new(None),
            custom_close: RefCell::new(None),
            custom_open: RefCell::new(None),
        }
    }

    pub fn get_from_frame<'a>(frame: &'a mut GlesFrame<'_, '_>) -> &'a Self {
        let data = frame.egl_context().user_data();
        data.get()
            .expect("shaders::init() must be called when creating the renderer")
    }

    pub fn get(renderer: &mut impl NiriRenderer) -> Option<&Self> {
        // Borrow-checker friendly two-step: probe the variant before taking the long-lived
        // borrow.
        if renderer.as_gles_renderer().is_some() {
            let renderer = renderer.as_gles_renderer().unwrap();
            let data = renderer.egl_context().user_data();
            Some(
                data.get()
                    .expect("shaders::init() must be called when creating the renderer"),
            )
        } else if renderer.as_vulkan_renderer().is_some() {
            let renderer = renderer.as_vulkan_renderer().unwrap();
            renderer.user_data().get()
        } else {
            None
        }
    }

    pub fn replace_custom_resize_program(
        &self,
        program: Option<ShaderProgram>,
    ) -> Option<ShaderProgram> {
        self.custom_resize.replace(program)
    }

    pub fn replace_custom_close_program(
        &self,
        program: Option<ShaderProgram>,
    ) -> Option<ShaderProgram> {
        self.custom_close.replace(program)
    }

    pub fn replace_custom_open_program(
        &self,
        program: Option<ShaderProgram>,
    ) -> Option<ShaderProgram> {
        self.custom_open.replace(program)
    }

    pub fn program(&self, program: ProgramType) -> Option<ShaderProgram> {
        match program {
            ProgramType::Border => self.border.clone(),
            ProgramType::Shadow => self.shadow.clone(),
            ProgramType::Resize => self
                .custom_resize
                .borrow()
                .clone()
                .or_else(|| self.resize.clone()),
            ProgramType::Close => self.custom_close.borrow().clone(),
            ProgramType::Open => self.custom_open.borrow().clone(),
        }
    }
}

/// Maps a GLES uniform declaration to a custom Vulkan uniform declaration.
fn uniform_name_to_decl(uniform: &UniformName<'_>) -> Option<CustomUniformDecl> {
    let kind = match uniform.type_ {
        UniformType::_1f => CustomUniformKind::Float,
        UniformType::_2f => CustomUniformKind::Vec2,
        UniformType::_3f => CustomUniformKind::Vec3,
        UniformType::_4f => CustomUniformKind::Vec4,
        UniformType::Matrix3x3 => CustomUniformKind::Mat3,
        _ => return None,
    };
    Some(CustomUniformDecl {
        name: uniform.name.clone().into_owned(),
        kind,
    })
}

/// Transforms a GLES-dialect niri fragment shader into Vulkan GLSL.
///
/// Uniform, varying and precision declarations are stripped from the source; uniforms are
/// provided through the generated std140 block, `niri_v_coords` becomes the vertex input,
/// and `niri_alpha`/`niri_tint` map to the push constants.
fn vulkanize_fragment(src: &str, decls: &[CustomUniformDecl], textures: &[&str]) -> String {
    let mut out = String::from(
        "#version 450
         #define DEBUG_FLAGS
         #define texture2D texture
",
    );
    out.push_str(&texture_bindings_glsl(textures));
    out.push_str(&uniform_block_glsl(decls));
    out.push_str(
        "layout(location = 0) in vec2 niri_v_coords;
         layout(location = 0) out vec4 niri_frag_color;
         #define gl_FragColor niri_frag_color
         layout(push_constant) uniform NiriPush {
             vec4 niri_pc0; vec4 niri_pc1; vec4 niri_pc2; vec4 niri_pc3; vec4 niri_pc4; vec4 niri_pc5;
         };
         #define niri_alpha niri_pc2.z
         #define niri_tint niri_pc2.w
         #define v_coords niri_v_coords
",
    );
    for line in src.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("#version")
            || trimmed.starts_with("#extension")
            || trimmed.starts_with("precision ")
            || trimmed.starts_with("varying ")
            || (trimmed.starts_with("uniform ") && trimmed.trim_end().ends_with(';'))
        {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Compiles a niri shader program for the Vulkan renderer from its GLES-dialect source.
pub(super) fn compile_vulkan_program(
    renderer: &mut VulkanRenderer,
    src: &str,
    uniforms: &[UniformName<'_>],
    texture_uniforms: &[&str],
) -> anyhow::Result<VulkanPixelProgram> {
    let mut decls: Vec<CustomUniformDecl> =
        uniforms.iter().filter_map(uniform_name_to_decl).collect();
    if !decls.iter().any(|d| d.name == "niri_size") {
        decls.push(CustomUniformDecl {
            name: "niri_size".to_owned(),
            kind: CustomUniformKind::Vec2,
        });
    }
    if !decls.iter().any(|d| d.name == "niri_scale") {
        decls.push(CustomUniformDecl {
            name: "niri_scale".to_owned(),
            kind: CustomUniformKind::Float,
        });
    }

    let vulkan_src = vulkanize_fragment(src, &decls, texture_uniforms);
    renderer
        .compile_custom_pixel_shader(&vulkan_src, &decls, texture_uniforms)
        .map_err(|err| {
            if std::env::var_os("NIRI_DUMP_SHADERS").is_some() {
                for (i, line) in vulkan_src.lines().enumerate() {
                    eprintln!("{:4} {line}", i + 1);
                }
            }
            anyhow::anyhow!("error compiling vulkan shader: {err}")
        })
}

impl Shaders {
    fn compile_vulkan(renderer: &mut VulkanRenderer) -> Self {
        let _span = tracy_client::span!("Shaders::compile_vulkan");

        let border = ShaderProgram::compile_vulkan(
            renderer,
            concat!(
                include_str!("border.frag"),
                include_str!("rounding_alpha.frag")
            ),
            &[
                UniformName::new("colorspace", UniformType::_1f),
                UniformName::new("hue_interpolation", UniformType::_1f),
                UniformName::new("color_from", UniformType::_4f),
                UniformName::new("color_to", UniformType::_4f),
                UniformName::new("grad_offset", UniformType::_2f),
                UniformName::new("grad_width", UniformType::_1f),
                UniformName::new("grad_vec", UniformType::_2f),
                UniformName::new("input_to_geo", UniformType::Matrix3x3),
                UniformName::new("geo_size", UniformType::_2f),
                UniformName::new("outer_radius", UniformType::_4f),
                UniformName::new("border_width", UniformType::_1f),
            ],
            &[],
        )
        .map_err(|err| {
            warn!("error compiling vulkan border shader: {err:?}");
        })
        .ok();

        let shadow = ShaderProgram::compile_vulkan(
            renderer,
            concat!(
                include_str!("shadow.frag"),
                include_str!("rounding_alpha.frag")
            ),
            &[
                UniformName::new("shadow_color", UniformType::_4f),
                UniformName::new("sigma", UniformType::_1f),
                UniformName::new("input_to_geo", UniformType::Matrix3x3),
                UniformName::new("geo_size", UniformType::_2f),
                UniformName::new("corner_radius", UniformType::_4f),
                UniformName::new("window_input_to_geo", UniformType::Matrix3x3),
                UniformName::new("window_geo_size", UniformType::_2f),
                UniformName::new("window_corner_radius", UniformType::_4f),
            ],
            &[],
        )
        .map_err(|err| {
            warn!("error compiling vulkan shadow shader: {err:?}");
        })
        .ok();

        let clipped_surface = {
            let src = concat!(
                include_str!("clipped_surface.frag"),
                include_str!("rounding_alpha.frag"),
                include_str!("hdr.frag"),
                "\nvec4 postprocess(vec4 color) { return color; }",
            );
            let uniforms = with_blend_uniform_names(&[
                UniformName::new("niri_scale", UniformType::_1f),
                UniformName::new("geo_size", UniformType::_2f),
                UniformName::new("corner_radius", UniformType::_4f),
                UniformName::new("input_to_geo", UniformType::Matrix3x3),
                // The GLES texture shader interface, injected by the renderer per draw.
                UniformName::new("alpha", UniformType::_1f),
                UniformName::new("tint", UniformType::_1f),
            ]);
            compile_vulkan_program(renderer, src, &uniforms, &["tex"])
                .map_err(|err| {
                    warn!("error compiling vulkan clipped surface shader: {err:?}");
                })
                .ok()
                .map(NiriTexProgram::Vulkan)
        };

        let resize = {
            let program = assemble_resize_program(include_str!("resize.frag"));
            ShaderProgram::compile_vulkan(renderer, &program, &resize_uniforms(), RESIZE_TEXTURES)
                .map_err(|err| {
                    warn!("error compiling vulkan resize shader: {err:?}");
                })
                .ok()
        };

        Shaders {
            texture_hdr: None,
            texture_hdr_to_sdr: None,
            border,
            shadow,
            clipped_surface,
            postprocess_and_clip: None,
            resize,
            gradient_fade: None,
            blur: None,
            custom_resize: RefCell::new(None),
            custom_close: RefCell::new(None),
            custom_open: RefCell::new(None),
        }
    }
}

/// Compiles and stores the shaders for a Vulkan renderer.
pub fn init_vulkan(renderer: &mut VulkanRenderer) {
    let shaders = Shaders::compile_vulkan(renderer);
    renderer.user_data().insert_if_missing(|| shaders);
}

pub fn init(renderer: &mut GlesRenderer) {
    let shaders = Shaders::compile(renderer);
    let data = renderer.egl_context().user_data();
    if !data.insert_if_missing(|| shaders) {
        error!("shaders were already compiled");
    }
}

fn assemble_resize_program(src: &str) -> String {
    let mut program = include_str!("resize_prelude.frag").to_string();
    program.push_str(src);
    program.push_str(include_str!("resize_epilogue.frag"));
    program.push_str(include_str!("rounding_alpha.frag"));
    program
}

fn resize_uniforms() -> [UniformName<'static>; 10] {
    [
        UniformName::new("niri_input_to_curr_geo", UniformType::Matrix3x3),
        UniformName::new("niri_curr_geo_to_prev_geo", UniformType::Matrix3x3),
        UniformName::new("niri_curr_geo_to_next_geo", UniformType::Matrix3x3),
        UniformName::new("niri_curr_geo_size", UniformType::_2f),
        UniformName::new("niri_geo_to_tex_prev", UniformType::Matrix3x3),
        UniformName::new("niri_geo_to_tex_next", UniformType::Matrix3x3),
        UniformName::new("niri_progress", UniformType::_1f),
        UniformName::new("niri_clamped_progress", UniformType::_1f),
        UniformName::new("niri_corner_radius", UniformType::_4f),
        UniformName::new("niri_clip_to_geometry", UniformType::_1f),
    ]
}

const RESIZE_TEXTURES: &[&str] = &["niri_tex_prev", "niri_tex_next"];

fn compile_resize_program(
    renderer: &mut impl NiriRenderer,
    src: &str,
) -> anyhow::Result<ShaderProgram> {
    let program = assemble_resize_program(src);
    let uniforms = &resize_uniforms();
    let textures = RESIZE_TEXTURES;

    if renderer.as_gles_renderer().is_some() {
        let renderer = renderer.as_gles_renderer().unwrap();
        Ok(ShaderProgram::compile(
            renderer, &program, uniforms, textures,
        )?)
    } else if renderer.as_vulkan_renderer().is_some() {
        let renderer = renderer.as_vulkan_renderer().unwrap();
        ShaderProgram::compile_vulkan(renderer, &program, uniforms, textures)
    } else {
        anyhow::bail!("unsupported renderer")
    }
}

pub fn set_custom_resize_program(renderer: &mut impl NiriRenderer, src: Option<&str>) {
    let program = if let Some(src) = src {
        match compile_resize_program(renderer, src) {
            Ok(program) => Some(program),
            Err(err) => {
                warn!("error compiling custom resize shader: {err:?}");
                return;
            }
        }
    } else {
        None
    };

    if let Some(prev) =
        Shaders::get(renderer).and_then(|s| s.replace_custom_resize_program(program))
    {
        if let Some(gles_renderer) = renderer.as_gles_renderer() {
            if let Err(err) = prev.destroy(gles_renderer) {
                warn!("error destroying previous custom resize shader: {err:?}");
            }
        }
    }
}

fn compile_close_program(
    renderer: &mut impl NiriRenderer,
    src: &str,
) -> anyhow::Result<ShaderProgram> {
    let mut program = include_str!("close_prelude.frag").to_string();
    program.push_str(src);
    program.push_str(include_str!("close_epilogue.frag"));

    let uniforms = &[
        UniformName::new("niri_input_to_geo", UniformType::Matrix3x3),
        UniformName::new("niri_geo_size", UniformType::_2f),
        UniformName::new("niri_geo_to_tex", UniformType::Matrix3x3),
        UniformName::new("niri_progress", UniformType::_1f),
        UniformName::new("niri_clamped_progress", UniformType::_1f),
        UniformName::new("niri_random_seed", UniformType::_1f),
    ];
    let textures: &[&str] = &["niri_tex"];

    if renderer.as_gles_renderer().is_some() {
        let renderer = renderer.as_gles_renderer().unwrap();
        Ok(ShaderProgram::compile(
            renderer, &program, uniforms, textures,
        )?)
    } else if renderer.as_vulkan_renderer().is_some() {
        let renderer = renderer.as_vulkan_renderer().unwrap();
        ShaderProgram::compile_vulkan(renderer, &program, uniforms, textures)
    } else {
        anyhow::bail!("unsupported renderer")
    }
}

pub fn set_custom_close_program(renderer: &mut impl NiriRenderer, src: Option<&str>) {
    let program = if let Some(src) = src {
        match compile_close_program(renderer, src) {
            Ok(program) => Some(program),
            Err(err) => {
                warn!("error compiling custom close shader: {err:?}");
                return;
            }
        }
    } else {
        None
    };

    if let Some(prev) = Shaders::get(renderer).and_then(|s| s.replace_custom_close_program(program))
    {
        if let Some(gles_renderer) = renderer.as_gles_renderer() {
            if let Err(err) = prev.destroy(gles_renderer) {
                warn!("error destroying previous custom close shader: {err:?}");
            }
        }
    }
}

fn compile_open_program(
    renderer: &mut impl NiriRenderer,
    src: &str,
) -> anyhow::Result<ShaderProgram> {
    let mut program = include_str!("open_prelude.frag").to_string();
    program.push_str(src);
    program.push_str(include_str!("open_epilogue.frag"));

    let uniforms = &[
        UniformName::new("niri_input_to_geo", UniformType::Matrix3x3),
        UniformName::new("niri_geo_size", UniformType::_2f),
        UniformName::new("niri_geo_to_tex", UniformType::Matrix3x3),
        UniformName::new("niri_progress", UniformType::_1f),
        UniformName::new("niri_clamped_progress", UniformType::_1f),
        UniformName::new("niri_random_seed", UniformType::_1f),
    ];
    let textures: &[&str] = &["niri_tex"];

    if renderer.as_gles_renderer().is_some() {
        let renderer = renderer.as_gles_renderer().unwrap();
        Ok(ShaderProgram::compile(
            renderer, &program, uniforms, textures,
        )?)
    } else if renderer.as_vulkan_renderer().is_some() {
        let renderer = renderer.as_vulkan_renderer().unwrap();
        ShaderProgram::compile_vulkan(renderer, &program, uniforms, textures)
    } else {
        anyhow::bail!("unsupported renderer")
    }
}

pub fn set_custom_open_program(renderer: &mut impl NiriRenderer, src: Option<&str>) {
    let program = if let Some(src) = src {
        match compile_open_program(renderer, src) {
            Ok(program) => Some(program),
            Err(err) => {
                warn!("error compiling custom open shader: {err:?}");
                return;
            }
        }
    } else {
        None
    };

    if let Some(prev) = Shaders::get(renderer).and_then(|s| s.replace_custom_open_program(program))
    {
        if let Some(gles_renderer) = renderer.as_gles_renderer() {
            if let Err(err) = prev.destroy(gles_renderer) {
                warn!("error destroying previous custom open shader: {err:?}");
            }
        }
    }
}

pub fn mat3_uniform(name: &str, mat: Mat3) -> Uniform<'_> {
    Uniform::new(
        name,
        UniformValue::Matrix3x3 {
            matrices: vec![mat.to_cols_array()],
            transpose: false,
        },
    )
}

#[cfg(test)]
mod tests {
    use smithay::backend::vulkan::{version::Version, Instance, PhysicalDevice};

    use super::*;

    #[test]
    fn vulkan_shaders_compile() {
        // Requires a Vulkan 1.3 device; skip silently when unavailable.
        let Ok(instance) = Instance::new(Version::VERSION_1_3, None) else {
            return;
        };
        let Ok(devices) = PhysicalDevice::enumerate(&instance) else {
            return;
        };
        let Some(mut renderer) = devices
            .into_iter()
            .find_map(|phd| smithay::backend::renderer::vulkan::VulkanRenderer::new(&phd).ok())
        else {
            return;
        };

        let shaders = Shaders::compile_vulkan(&mut renderer);
        assert!(
            shaders.border.is_some(),
            "vulkan border shader failed to compile"
        );
        assert!(
            shaders.shadow.is_some(),
            "vulkan shadow shader failed to compile"
        );
        assert!(
            shaders.clipped_surface.is_some(),
            "vulkan clipped surface shader failed to compile"
        );
        assert!(
            shaders.resize.is_some(),
            "vulkan resize shader failed to compile"
        );

        // Representative user custom shaders must keep working through the transformer.
        let mut resize = include_str!("resize_prelude.frag").to_string();
        resize.push_str(
            "vec4 resize_color(vec3 coords_curr_geo, vec3 size_curr_geo) {\n\
                 vec3 coords = niri_geo_to_tex_next * coords_curr_geo;\n\
                 vec4 color = texture2D(niri_tex_next, coords.st);\n\
                 return color * niri_clamped_progress;\n\
             }\n",
        );
        resize.push_str(include_str!("resize_epilogue.frag"));
        resize.push_str(include_str!("rounding_alpha.frag"));
        let uniforms = [
            UniformName::new("niri_input_to_curr_geo", UniformType::Matrix3x3),
            UniformName::new("niri_curr_geo_to_prev_geo", UniformType::Matrix3x3),
            UniformName::new("niri_curr_geo_to_next_geo", UniformType::Matrix3x3),
            UniformName::new("niri_curr_geo_size", UniformType::_2f),
            UniformName::new("niri_geo_to_tex_prev", UniformType::Matrix3x3),
            UniformName::new("niri_geo_to_tex_next", UniformType::Matrix3x3),
            UniformName::new("niri_progress", UniformType::_1f),
            UniformName::new("niri_clamped_progress", UniformType::_1f),
            UniformName::new("niri_corner_radius", UniformType::_4f),
            UniformName::new("niri_clip_to_geometry", UniformType::_1f),
        ];
        ShaderProgram::compile_vulkan(
            &mut renderer,
            &resize,
            &uniforms,
            &["niri_tex_prev", "niri_tex_next"],
        )
        .expect("vulkan resize example shader failed to compile");

        let mut close = include_str!("close_prelude.frag").to_string();
        close.push_str(
            "vec4 close_color(vec3 coords_geo, vec3 size_geo) {\n\
                 vec3 coords = niri_geo_to_tex * coords_geo;\n\
                 vec4 color = texture2D(niri_tex, coords.st);\n\
                 return color * (1.0 - niri_clamped_progress);\n\
             }\n",
        );
        close.push_str(include_str!("close_epilogue.frag"));
        let uniforms = [
            UniformName::new("niri_input_to_geo", UniformType::Matrix3x3),
            UniformName::new("niri_geo_size", UniformType::_2f),
            UniformName::new("niri_geo_to_tex", UniformType::Matrix3x3),
            UniformName::new("niri_progress", UniformType::_1f),
            UniformName::new("niri_clamped_progress", UniformType::_1f),
            UniformName::new("niri_random_seed", UniformType::_1f),
        ];
        ShaderProgram::compile_vulkan(&mut renderer, &close, &uniforms, &["niri_tex"])
            .expect("vulkan close example shader failed to compile");
    }
}
