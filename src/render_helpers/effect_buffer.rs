use std::mem;

use anyhow::{ensure, Context as _};
use smithay::backend::allocator::Fourcc;
use smithay::backend::renderer::damage::OutputDamageTracker;
use smithay::backend::renderer::element::{Id, RenderElement, RenderElementStates};
use smithay::backend::renderer::gles::{GlesFrame, GlesRenderer, GlesTexture};
use smithay::backend::renderer::utils::CommitCounter;
use smithay::backend::renderer::vulkan::{VulkanFrame, VulkanRenderer, VulkanTexture};
use smithay::backend::renderer::{
    Color32F, ContextId, ErasedContextId, FrameContext as _, Offscreen as _, Renderer as _, Texture,
};
use smithay::utils::{Buffer, Logical, Physical, Scale, Size, Transform};

use crate::backend::tty_renderer::TtyOffscreen;
use crate::layer::mapped::LayerSurfaceRenderElement;
use crate::render_helpers::blur::{Blur, BlurOptions, VulkanBlur};
use crate::render_helpers::renderer::{NiriCaptureRenderer, NiriRenderer};

#[derive(Debug)]
pub struct EffectBuffer {
    /// Id to be used for this effect buffer's elements.
    id: Id,

    /// Size of the effect buffer.
    size: Size<i32, Buffer>,
    /// Scale of the effect buffer.
    scale: Scale<f64>,
    /// Options for blurring.
    blur_options: BlurOptions,

    /// Elements to be rendered on demand.
    elements: Elements,
    /// Offscreen buffer where elements get rendered.
    offscreen: Option<Offscreen>,
    /// Blurring program, if available.
    blur: Option<BlurVariant>,

    /// Commit counter that takes into account both original and blurred texture changes.
    commit_counter: CommitCounter,
}

#[derive(Debug)]
enum Elements {
    /// Contents remain unchanged.
    Unchanged(ElementStore),
    /// New contents, need to check damage and render.
    New(ElementStore),
}

/// Per-frame xray contents, typed for the raw renderer that created them.
///
/// The TTY renderer enum borrows the GPU manager and cannot be stored, so contents are
/// captured with the raw primary renderer (which is `'static`), like the GLES path always
/// did.
#[derive(Debug)]
pub enum ElementStore {
    Gles(Vec<LayerSurfaceRenderElement<GlesRenderer>>),
    Vulkan(Vec<LayerSurfaceRenderElement<VulkanRenderer>>),
}

impl ElementStore {
    fn clear(&mut self) {
        match self {
            ElementStore::Gles(elements) => elements.clear(),
            ElementStore::Vulkan(elements) => elements.clear(),
        }
    }
}

/// Accessor for the [`ElementStore`] variant of a raw renderer.
pub trait XrayElementStore: NiriRenderer + Sized {
    fn elements(store: &mut ElementStore) -> &mut Vec<LayerSurfaceRenderElement<Self>>;
}

impl XrayElementStore for GlesRenderer {
    fn elements(store: &mut ElementStore) -> &mut Vec<LayerSurfaceRenderElement<Self>> {
        if !matches!(store, ElementStore::Gles(_)) {
            *store = ElementStore::Gles(Vec::new());
        }
        let ElementStore::Gles(elements) = store else {
            unreachable!();
        };
        elements
    }
}

impl XrayElementStore for VulkanRenderer {
    fn elements(store: &mut ElementStore) -> &mut Vec<LayerSurfaceRenderElement<Self>> {
        if !matches!(store, ElementStore::Vulkan(_)) {
            *store = ElementStore::Vulkan(Vec::new());
        }
        let ElementStore::Vulkan(elements) = store else {
            unreachable!();
        };
        elements
    }
}

#[derive(Debug)]
enum BlurVariant {
    Gles(Blur),
    Vulkan(VulkanBlur),
}

#[derive(Debug)]
struct Offscreen {
    /// The texture with the offscreen contents.
    texture: TtyOffscreen,
    /// Id of the renderer context that the texture comes from.
    renderer_context_id: ErasedContextId,
    /// Scale of the texture.
    scale: Scale<f64>,
    /// Damage tracker for drawing to the texture.
    damage: OutputDamageTracker,
    /// Render element states from the last render into the offscreen.
    states: RenderElementStates,
    /// Rendered blurred version of the texture.
    ///
    /// When texture needs to be reblurred, this field must be reset to `None`.
    blurred: Option<TtyOffscreen>,
}

impl Default for Elements {
    fn default() -> Self {
        Self::Unchanged(ElementStore::Gles(Vec::new()))
    }
}

impl EffectBuffer {
    pub fn new() -> Self {
        Self {
            id: Id::new(),
            size: Size::default(),
            scale: Scale::from(1.),
            blur_options: BlurOptions::default(),
            elements: Elements::default(),
            offscreen: None,
            blur: None,
            commit_counter: CommitCounter::default(),
        }
    }

    pub fn id(&self) -> &Id {
        &self.id
    }

    pub fn commit(&self) -> CommitCounter {
        self.commit_counter
    }

    pub fn logical_size(&self) -> Size<f64, Logical> {
        self.size.to_f64().to_logical(self.scale, Transform::Normal)
    }

    pub fn scale(&self) -> Scale<f64> {
        self.scale
    }

    pub fn render_element_states(&self) -> Option<&RenderElementStates> {
        self.offscreen.as_ref().map(|o| &o.states)
    }

    pub fn update_size(&mut self, size: Size<i32, Physical>, scale: Scale<f64>) {
        self.size = size.to_logical(1).to_buffer(1, Transform::Normal);
        self.scale = scale;
    }

    pub fn update_blur_options(&mut self, options: BlurOptions) {
        if self.blur_options == options {
            return;
        }

        self.blur_options = options;

        if let Some(offscreen) = &mut self.offscreen {
            if offscreen.blurred.is_some() {
                offscreen.blurred = None;
                self.commit_counter.increment();
            }
        }
    }

    pub fn elements<R: XrayElementStore>(&mut self) -> &mut Vec<LayerSurfaceRenderElement<R>> {
        // Assume we're going to insert new elements, switch to New.
        match mem::take(&mut self.elements) {
            Elements::Unchanged(elements) | Elements::New(elements) => {
                self.elements = Elements::New(elements);
            }
        }
        let Elements::New(store) = &mut self.elements else {
            unreachable!();
        };
        R::elements(store)
    }

    pub fn clear_elements(&mut self) {
        match mem::take(&mut self.elements) {
            Elements::Unchanged(mut store) | Elements::New(mut store) => {
                store.clear();
                self.elements = Elements::New(store);
            }
        }
    }

    /// Prepares the offscreen (and optionally blurred) contents, dispatching to the raw
    /// renderer that filled the elements.
    pub fn prepare(&mut self, renderer: &mut impl NiriRenderer, blur: bool) -> bool {
        // Borrow-checker friendly two-step: probe the variant before taking the long-lived
        // borrow.
        if renderer.as_gles_renderer().is_some() {
            let renderer = renderer.as_gles_renderer().unwrap();
            self.prepare_for(renderer, blur, Self::prepare_blur_gles)
        } else if renderer.as_vulkan_renderer().is_some() {
            let renderer = renderer.as_vulkan_renderer().unwrap();
            self.prepare_for(renderer, blur, Self::prepare_blur_vulkan)
        } else {
            false
        }
    }

    fn prepare_for<R>(
        &mut self,
        renderer: &mut R,
        blur: bool,
        prepare_blur: impl FnOnce(&mut Self, &mut R) -> anyhow::Result<()>,
    ) -> bool
    where
        R: NiriCaptureRenderer + XrayElementStore,
        R::Error: Send + Sync + 'static,
        LayerSurfaceRenderElement<R>: RenderElement<R>,
    {
        if let Err(err) = self.prepare_offscreen(renderer) {
            warn!("error preparing offscreen: {err:?}");
            return false;
        };

        if blur {
            if let Err(err) = prepare_blur(self, renderer) {
                warn!("error preparing blur: {err:?}");
                return false;
            }
        }

        true
    }

    fn prepare_offscreen<R>(&mut self, renderer: &mut R) -> anyhow::Result<()>
    where
        R: NiriCaptureRenderer + XrayElementStore,
        R::Error: Send + Sync + 'static,
        LayerSurfaceRenderElement<R>: RenderElement<R>,
    {
        let _span = tracy_client::span!("EffectBuffer::prepare_offscreen");

        // Check if we need to create or recreate the texture.
        let size_string;
        let mut reason = "";
        if let Some(Offscreen {
            texture,
            renderer_context_id,
            ..
        }) = &mut self.offscreen
        {
            let old_size = texture.size();
            if old_size != self.size {
                size_string = format!(
                    "size changed from {} × {} to {} × {}",
                    old_size.w, old_size.h, self.size.w, self.size.h
                );
                reason = &size_string;

                self.offscreen = None;
            } else if !texture.is_unique_reference() {
                reason = "not unique";

                self.offscreen = None;
            } else if *renderer_context_id != ContextId::erased(&renderer.context_id()) {
                reason = "renderer id changed";

                self.offscreen = None;
            }
        } else {
            reason = "first render";
        }

        let offscreen = if let Some(offscreen) = &mut self.offscreen {
            offscreen
        } else {
            trace!("creating new offscreen texture: {reason}");
            let span = tracy_client::span!("creating effect offscreen texture");
            span.emit_text(reason);

            let texture = renderer
                .create_buffer(Fourcc::Abgr8888, self.size)
                .context("error creating texture")?;

            let buffer_size = self.size.to_logical(1, Transform::Normal).to_physical(1);
            let damage = OutputDamageTracker::new(buffer_size, self.scale, Transform::Normal);

            self.offscreen.insert(Offscreen {
                texture: R::wrap_offscreen(texture),
                renderer_context_id: ContextId::erased(&renderer.context_id()),
                scale: self.scale,
                damage,
                states: RenderElementStates::default(),
                blurred: None,
            })
        };

        // Recreate the damage tracker if the scale changes. We already recreate it for buffer size
        // changes, and transform is always Normal.
        if offscreen.scale != self.scale {
            offscreen.scale = self.scale;

            trace!("recreating damage tracker due to scale change");
            let buffer_size = self.size.to_logical(1, Transform::Normal).to_physical(1);
            offscreen.damage = OutputDamageTracker::new(buffer_size, self.scale, Transform::Normal);

            self.commit_counter.increment();
            offscreen.blurred = None;
        }

        // Render the elements if any.
        let mut store = match mem::take(&mut self.elements) {
            Elements::New(store) => store,
            x @ Elements::Unchanged(_) => {
                // No redrawing necessary.
                self.elements = x;
                return Ok(());
            }
        };
        let elements = R::elements(&mut store);

        let res = {
            let texture = R::unwrap_offscreen(&mut offscreen.texture)
                .context("offscreen texture is from a different renderer")?;
            let mut target = renderer.bind(texture).context("error binding texture")?;
            offscreen
                .damage
                .render_output(renderer, &mut target, 1, elements, Color32F::TRANSPARENT)
                .context("error rendering")?
        };

        offscreen.states = res.states;

        if res.damage.is_some() {
            self.commit_counter.increment();

            // Original texture changed; reset the blurred texture.
            offscreen.blurred = None;
        }

        // Clear and put the storage back.
        elements.clear();
        self.elements = Elements::Unchanged(store);

        Ok(())
    }

    fn prepare_blur_gles(&mut self, renderer: &mut GlesRenderer) -> anyhow::Result<()> {
        let offscreen = self.offscreen.as_mut().context("missing offscreen")?;
        if offscreen.blurred.is_some() {
            // Already rendered.
            return Ok(());
        }

        if let Some(BlurVariant::Gles(blur)) = &self.blur {
            if blur.context_id() != renderer.context_id() {
                debug!("recreating blur: renderer changed");
                self.blur = None;
            }
        } else if self.blur.is_some() {
            debug!("recreating blur: renderer variant changed");
            self.blur = None;
        }

        let blur = if let Some(BlurVariant::Gles(blur)) = &mut self.blur {
            blur
        } else {
            let Some(blur) = Blur::new(renderer) else {
                // Missing blur shader.
                return Ok(());
            };
            let BlurVariant::Gles(blur) = self.blur.insert(BlurVariant::Gles(blur)) else {
                unreachable!();
            };
            blur
        };

        ensure!(
            offscreen.renderer_context_id == ContextId::erased(&renderer.context_id()),
            "wrong renderer context id"
        );

        let TtyOffscreen::Gles(texture) = &offscreen.texture else {
            anyhow::bail!("offscreen texture is not a GLES texture");
        };

        blur.prepare_textures(
            |fourcc, size| renderer.create_buffer(fourcc, size),
            texture,
            self.blur_options,
        )
        .context("error preparing blur textures")?;

        Ok(())
    }

    fn prepare_blur_vulkan(&mut self, renderer: &mut VulkanRenderer) -> anyhow::Result<()> {
        let offscreen = self.offscreen.as_mut().context("missing offscreen")?;
        if offscreen.blurred.is_some() {
            // Already rendered.
            return Ok(());
        }

        if self.blur.is_some() && !matches!(self.blur, Some(BlurVariant::Vulkan(_))) {
            debug!("recreating blur: renderer variant changed");
            self.blur = None;
        }

        let blur = if let Some(BlurVariant::Vulkan(blur)) = &mut self.blur {
            blur
        } else {
            let Some(blur) = VulkanBlur::new(renderer) else {
                // Missing blur shader.
                return Ok(());
            };
            let BlurVariant::Vulkan(blur) = self.blur.insert(BlurVariant::Vulkan(blur)) else {
                unreachable!();
            };
            blur
        };

        ensure!(
            offscreen.renderer_context_id == ContextId::erased(&renderer.context_id()),
            "wrong renderer context id"
        );

        let TtyOffscreen::Vulkan(texture) = &offscreen.texture else {
            anyhow::bail!("offscreen texture is not a Vulkan texture");
        };

        blur.prepare_textures(renderer, texture, self.blur_options)
            .context("error preparing blur textures")?;

        Ok(())
    }

    pub fn render_gles(
        &mut self,
        frame: &mut GlesFrame,
        blur: bool,
    ) -> anyhow::Result<GlesTexture> {
        let offscreen = self.offscreen.as_mut().context("offscreen is missing")?;

        let TtyOffscreen::Gles(texture) = &offscreen.texture else {
            anyhow::bail!("offscreen texture is not a GLES texture");
        };

        if !blur {
            return Ok(texture.clone());
        }

        if let Some(TtyOffscreen::Gles(texture)) = &offscreen.blurred {
            return Ok(texture.clone());
        }

        let Some(BlurVariant::Gles(blur)) = &mut self.blur else {
            anyhow::bail!("blur is missing");
        };
        let mut guard = frame.renderer();
        let renderer = guard.as_mut();
        let blurred = blur
            .render(renderer, texture, self.blur_options)
            .context("error rendering blur")?;
        offscreen.blurred = Some(TtyOffscreen::Gles(blurred.clone()));

        Ok(blurred)
    }

    pub fn render_vulkan(
        &mut self,
        frame: &mut VulkanFrame<'_, '_>,
        blur: bool,
    ) -> anyhow::Result<VulkanTexture> {
        let offscreen = self.offscreen.as_mut().context("offscreen is missing")?;

        let TtyOffscreen::Vulkan(texture) = &offscreen.texture else {
            anyhow::bail!("offscreen texture is not a Vulkan texture");
        };

        if !blur {
            return Ok(texture.clone());
        }

        if let Some(TtyOffscreen::Vulkan(texture)) = &offscreen.blurred {
            return Ok(texture.clone());
        }

        let Some(BlurVariant::Vulkan(blur)) = &mut self.blur else {
            anyhow::bail!("blur is missing");
        };
        let blurred = blur
            .render(frame, texture, self.blur_options)
            .context("error rendering blur")?;
        offscreen.blurred = Some(TtyOffscreen::Vulkan(blurred.clone()));

        Ok(blurred)
    }
}
