use smithay::backend::renderer::element::{Element, Id, Kind, RenderElement, UnderlyingStorage};
use smithay::backend::renderer::gles::{GlesError, GlesFrame, GlesRenderer, Uniform};
use smithay::backend::renderer::utils::{CommitCounter, DamageSet, OpaqueRegions};
use smithay::utils::user_data::UserDataMap;
use smithay::utils::{Buffer, Physical, Rectangle, Scale, Transform};

use super::blend::FrameBlendState;
use super::texture::{TextureRenderElement, UniversalTextureRenderElement};
use crate::backend::tty::{TtyFrame, TtyRenderer, TtyRendererError};
use crate::backend::tty_renderer::TtyOffscreen;
use crate::render_helpers::renderer::{AsGlesFrame as _, NiriRenderer};
use crate::render_helpers::shaders::{NiriTexProgram, Shaders};

#[derive(Debug, Clone)]
pub struct GradientFadeTextureRenderElement {
    inner: UniversalTextureRenderElement,
    program: GradientFadeShader,
    cutoff: (f32, f32),
}

#[derive(Debug, Clone)]
pub struct GradientFadeShader(NiriTexProgram);

impl GradientFadeTextureRenderElement {
    pub fn new(texture: TextureRenderElement<TtyOffscreen>, program: GradientFadeShader) -> Self {
        let logical_w = texture.buffer().logical_size().w;
        let logical_src_w = texture.logical_src().size.w;
        let cutoff = if logical_src_w < logical_w {
            // Texture is clipped, add a fade.
            let cutoff = 1. - f64::min(18. / logical_src_w, 1.);
            let full = logical_src_w / logical_w;
            ((cutoff * full) as f32, full as f32)
        } else {
            // Texture is displayed full-size, no cutoff necessary.
            (1., 1.)
        };
        Self {
            inner: UniversalTextureRenderElement(texture),
            program,
            cutoff,
        }
    }

    pub fn shader<R: NiriRenderer>(renderer: &mut R) -> Option<GradientFadeShader> {
        let program = Shaders::get(renderer).and_then(|s| s.gradient_fade.clone());
        program.map(GradientFadeShader)
    }
}

impl Element for GradientFadeTextureRenderElement {
    fn id(&self) -> &Id {
        self.inner.id()
    }

    fn current_commit(&self) -> CommitCounter {
        self.inner.current_commit()
    }

    fn geometry(&self, scale: Scale<f64>) -> Rectangle<i32, Physical> {
        self.inner.geometry(scale)
    }

    fn transform(&self) -> Transform {
        self.inner.transform()
    }

    fn src(&self) -> Rectangle<f64, Buffer> {
        self.inner.src()
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

impl RenderElement<GlesRenderer> for GradientFadeTextureRenderElement {
    fn draw(
        &self,
        frame: &mut GlesFrame<'_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        cache: Option<&UserDataMap>,
    ) -> Result<(), GlesError> {
        let NiriTexProgram::Gles(program) = &self.program.0 else {
            return Ok(());
        };
        let mut uniforms = vec![Uniform::new("cutoff", self.cutoff)];
        uniforms.extend(FrameBlendState::uniforms(frame));
        let saved = frame.take_tex_program_override();
        frame.override_default_tex_program(program.clone(), uniforms);
        let res = RenderElement::<GlesRenderer>::draw(
            &self.inner,
            frame,
            src,
            dst,
            damage,
            opaque_regions,
            cache,
        );
        frame.set_tex_program_override(saved);
        res
    }

    fn underlying_storage(&self, _renderer: &mut GlesRenderer) -> Option<UnderlyingStorage<'_>> {
        // If scanout for things other than Wayland buffers is implemented, this will need to take
        // the target GPU into account.
        None
    }
}

impl<'render> RenderElement<TtyRenderer<'render>> for GradientFadeTextureRenderElement {
    fn draw(
        &self,
        frame: &mut TtyFrame<'render, '_, '_>,
        src: Rectangle<f64, Buffer>,
        dst: Rectangle<i32, Physical>,
        damage: &[Rectangle<i32, Physical>],
        opaque_regions: &[Rectangle<i32, Physical>],
        cache: Option<&UserDataMap>,
    ) -> Result<(), TtyRendererError<'render>> {
        let mut saved = None;
        let mut saved_vk = None;
        match (&self.program.0, &mut *frame) {
            (NiriTexProgram::Gles(program), _) => {
                if let Some(gles_frame) = frame.as_gles_frame() {
                    let mut uniforms = vec![Uniform::new("cutoff", self.cutoff)];
                    uniforms.extend(FrameBlendState::uniforms(gles_frame));
                    saved = Some(gles_frame.take_tex_program_override());
                    gles_frame.override_default_tex_program(program.clone(), uniforms);
                }
            }
            (NiriTexProgram::Vulkan(program), TtyFrame::Vulkan(multi)) => {
                let vk_frame: &mut smithay::backend::renderer::vulkan::VulkanFrame<'_, '_> =
                    multi.as_mut();
                let uniforms_src = vec![Uniform::new("cutoff", self.cutoff)];
                let mut uniforms: Vec<_> = uniforms_src
                    .iter()
                    .filter_map(super::shader_element::uniform_to_custom_owned)
                    .collect();
                uniforms.extend(
                    super::blend::vulkan_blend_custom_uniforms(
                        vk_frame,
                        super::blend::ContentColor::default(),
                    )
                    .into_iter()
                    .map(|u| {
                        smithay::backend::renderer::vulkan::OwnedCustomUniform {
                            name: u.name.to_owned(),
                            value: u.value,
                        }
                    }),
                );
                saved_vk = Some(vk_frame.take_tex_program_override());
                vk_frame.set_tex_program_override(Some((program.clone(), uniforms)));
            }
            _ => {}
        }
        let res = RenderElement::<TtyRenderer>::draw(
            &self.inner,
            frame,
            src,
            dst,
            damage,
            opaque_regions,
            cache,
        );
        if let Some(saved) = saved {
            if let Some(gles_frame) = frame.as_gles_frame() {
                gles_frame.set_tex_program_override(saved);
            }
        }
        if let Some(saved) = saved_vk {
            if let TtyFrame::Vulkan(multi) = frame {
                let vk_frame: &mut smithay::backend::renderer::vulkan::VulkanFrame<'_, '_> =
                    multi.as_mut();
                vk_frame.set_tex_program_override(saved);
            }
        }
        res
    }

    fn underlying_storage(
        &self,
        _renderer: &mut TtyRenderer<'render>,
    ) -> Option<UnderlyingStorage<'_>> {
        // If scanout for things other than Wayland buffers is implemented, this will need to take
        // the target GPU into account.
        None
    }
}
