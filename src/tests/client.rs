use std::cmp::min;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as _;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use calloop::EventLoop;
use calloop_wayland_source::WaylandSource;
use single_pixel_buffer::v1::client::wp_single_pixel_buffer_manager_v1::WpSinglePixelBufferManagerV1;
use smithay::reexports::wayland_protocols::wp::color_management::v1::client::wp_color_management_output_v1::{self, WpColorManagementOutputV1};
use smithay::reexports::wayland_protocols::wp::color_management::v1::client::wp_color_management_surface_feedback_v1::{self, WpColorManagementSurfaceFeedbackV1};
use smithay::reexports::wayland_protocols::wp::color_management::v1::client::wp_color_management_surface_v1::WpColorManagementSurfaceV1;
use smithay::reexports::wayland_protocols::wp::color_management::v1::client::wp_color_manager_v1::{
    Primaries, RenderIntent, TransferFunction, WpColorManagerV1,
};
use smithay::reexports::wayland_protocols::wp::color_management::v1::client::wp_image_description_creator_params_v1::WpImageDescriptionCreatorParamsV1;
use smithay::reexports::wayland_protocols::wp::color_management::v1::client::wp_image_description_info_v1::{self, WpImageDescriptionInfoV1};
use smithay::reexports::wayland_protocols::wp::color_management::v1::client::wp_image_description_v1::{self, WpImageDescriptionV1};
use smithay::reexports::wayland_protocols::wp::single_pixel_buffer;
use smithay::reexports::wayland_protocols::wp::viewporter::client::wp_viewport::WpViewport;
use smithay::reexports::wayland_protocols::wp::viewporter::client::wp_viewporter::WpViewporter;
use smithay::reexports::wayland_protocols::xdg::decoration::zv1::client::zxdg_decoration_manager_v1::ZxdgDecorationManagerV1;
use smithay::reexports::wayland_protocols::xdg::decoration::zv1::client::zxdg_toplevel_decoration_v1::{
    self, ZxdgToplevelDecorationV1,
};
use smithay::reexports::wayland_protocols::xdg::shell::client::xdg_surface::{self, XdgSurface};
use smithay::reexports::wayland_protocols::xdg::shell::client::xdg_toplevel::{self, XdgToplevel};
use smithay::reexports::wayland_protocols::xdg::shell::client::xdg_wm_base::{self, XdgWmBase};
use smithay::reexports::wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_shell_v1::{
    self, ZwlrLayerShellV1,
};
use smithay::reexports::wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1::{
    self, ZwlrLayerSurfaceV1,
};
use smithay::reexports::wayland_protocols_wlr::virtual_pointer::v1::client::zwlr_virtual_pointer_manager_v1::ZwlrVirtualPointerManagerV1;
use smithay::reexports::wayland_protocols_wlr::virtual_pointer::v1::client::zwlr_virtual_pointer_v1::ZwlrVirtualPointerV1;
use wayland_backend::client::Backend;
use wayland_client::globals::Global;
use wayland_client::protocol::wl_buffer::{self, WlBuffer};
use wayland_client::protocol::wl_callback::{self, WlCallback};
use wayland_client::protocol::wl_compositor::WlCompositor;
use wayland_client::protocol::wl_display::WlDisplay;
use wayland_client::protocol::wl_output::{self, WlOutput};
use wayland_client::protocol::wl_registry::{self, WlRegistry};
use wayland_client::protocol::wl_subcompositor::WlSubcompositor;
use wayland_client::protocol::wl_subsurface::{self, WlSubsurface};
use wayland_client::protocol::wl_surface::{self, WlSurface};
use wayland_client::{Connection, Dispatch, Proxy as _, QueueHandle};

use crate::utils::id::IdCounter;

pub struct Client {
    pub id: ClientId,
    pub event_loop: EventLoop<'static, State>,
    pub connection: Connection,
    pub qh: QueueHandle<State>,
    pub display: WlDisplay,
    pub state: State,
}

pub struct State {
    pub qh: QueueHandle<State>,

    pub globals: Vec<Global>,
    pub outputs: HashMap<WlOutput, String>,

    pub compositor: Option<WlCompositor>,
    pub xdg_wm_base: Option<XdgWmBase>,
    pub layer_shell: Option<ZwlrLayerShellV1>,
    pub virtual_pointer_manager: Option<ZwlrVirtualPointerManagerV1>,
    pub spbm: Option<WpSinglePixelBufferManagerV1>,
    pub viewporter: Option<WpViewporter>,
    pub subcompositor: Option<WlSubcompositor>,
    pub decoration_manager: Option<ZxdgDecorationManagerV1>,
    pub color_manager: Option<WpColorManagerV1>,
    /// Feedback objects kept alive so preferred_changed events can arrive.
    pub surface_feedbacks: Vec<WpColorManagementSurfaceFeedbackV1>,
    /// Identities received in preferred_changed events, in order.
    pub preferred_changed: Vec<u32>,
    /// Identities received in wp_image_description_v1.ready events, in order.
    pub ready_identities: Vec<u32>,
    /// Named transfer function / primaries from the latest image description info exchange.
    pub info_tf: Option<TransferFunction>,
    pub info_primaries: Option<Primaries>,
    /// Last received luminances info event as (min ×10000, max, reference white).
    pub info_luminances: Option<(u32, u32, u32)>,
    /// Last received target_luminance info event as (min ×10000, max).
    pub info_target_luminance: Option<(u32, u32)>,

    pub windows: Vec<Window>,
    pub layers: Vec<LayerSurface>,
}

pub struct Window {
    pub qh: QueueHandle<State>,
    pub spbm: WpSinglePixelBufferManagerV1,

    pub surface: WlSurface,
    pub xdg_surface: XdgSurface,
    pub xdg_toplevel: XdgToplevel,
    pub viewport: WpViewport,
    pub decoration_manager: Option<ZxdgDecorationManagerV1>,
    pub decoration: Option<ZxdgToplevelDecorationV1>,
    pub pending_configure: Configure,
    pub configures_received: Vec<(u32, Configure)>,
    pub close_requested: bool,

    pub configures_looked_at: usize,
}

pub struct LayerSurface {
    pub qh: QueueHandle<State>,
    pub spbm: WpSinglePixelBufferManagerV1,

    pub surface: WlSurface,
    pub layer_surface: ZwlrLayerSurfaceV1,
    pub viewport: WpViewport,
    pub configures_received: Vec<(u32, LayerConfigure)>,
    pub close_requested: bool,

    pub configures_looked_at: usize,
}

#[derive(Debug, Clone, Default)]
pub struct Configure {
    pub size: (i32, i32),
    pub bounds: Option<(i32, i32)>,
    pub states: Vec<xdg_toplevel::State>,
    pub decoration_mode: Option<zxdg_toplevel_decoration_v1::Mode>,
}

#[derive(Debug, Clone, Copy)]
pub struct LayerConfigure {
    pub size: (u32, u32),
}

#[derive(Clone, Copy, Default)]
pub struct LayerMargin {
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
    pub left: i32,
}

#[derive(Clone, Copy, Default)]
pub struct LayerConfigureProps {
    pub size: Option<(u32, u32)>,
    pub anchor: Option<zwlr_layer_surface_v1::Anchor>,
    pub exclusive_zone: Option<i32>,
    pub margin: Option<LayerMargin>,
    pub kb_interactivity: Option<zwlr_layer_surface_v1::KeyboardInteractivity>,
    pub layer: Option<zwlr_layer_shell_v1::Layer>,
    pub exclusive_edge: Option<zwlr_layer_surface_v1::Anchor>,
}

#[derive(Default)]
pub struct SyncData {
    pub done: AtomicBool,
}

static CLIENT_ID_COUNTER: IdCounter = IdCounter::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClientId(u64);

impl ClientId {
    fn next() -> ClientId {
        ClientId(CLIENT_ID_COUNTER.next())
    }
}

impl fmt::Display for Configure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "size: {} × {}, ", self.size.0, self.size.1)?;
        if let Some(bounds) = self.bounds {
            write!(f, "bounds: {} × {}, ", bounds.0, bounds.1)?;
        } else {
            write!(f, "bounds: none, ")?;
        }
        write!(f, "states: {:?}", self.states)?;
        if let Some(mode) = self.decoration_mode {
            write!(f, ", decoration: {mode:?}")?;
        }
        Ok(())
    }
}

impl fmt::Display for LayerConfigure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "size: {} × {}", self.size.0, self.size.1)?;
        Ok(())
    }
}

impl Client {
    pub fn new(stream: UnixStream) -> Self {
        let id = ClientId::next();

        let event_loop = EventLoop::try_new().unwrap();
        let backend = Backend::connect(stream).unwrap();
        let connection = Connection::from_backend(backend);
        let queue = connection.new_event_queue();
        let qh = queue.handle();
        WaylandSource::new(connection.clone(), queue)
            .insert(event_loop.handle())
            .unwrap();

        let display = connection.display();
        let _registry = display.get_registry(&qh, ());
        connection.flush().unwrap();

        let state = State {
            qh: qh.clone(),
            globals: Vec::new(),
            outputs: HashMap::new(),
            compositor: None,
            xdg_wm_base: None,
            layer_shell: None,
            virtual_pointer_manager: None,
            spbm: None,
            viewporter: None,
            subcompositor: None,
            decoration_manager: None,
            color_manager: None,
            surface_feedbacks: Vec::new(),
            preferred_changed: Vec::new(),
            ready_identities: Vec::new(),
            info_tf: None,
            info_primaries: None,
            info_luminances: None,
            info_target_luminance: None,
            windows: Vec::new(),
            layers: Vec::new(),
        };

        Self {
            id,
            event_loop,
            connection,
            qh,
            display,
            state,
        }
    }

    pub fn dispatch(&mut self) {
        self.event_loop
            .dispatch(Duration::ZERO, &mut self.state)
            .unwrap();

        if let Some(error) = self.connection.protocol_error() {
            panic!("{error}");
        }
    }

    pub fn send_sync(&self) -> Arc<SyncData> {
        let data = Arc::new(SyncData::default());
        self.display.sync(&self.qh, data.clone());
        self.connection.flush().unwrap();
        data
    }

    pub fn create_window(&mut self) -> &mut Window {
        self.state.create_window()
    }

    pub fn window(&mut self, surface: &WlSurface) -> &mut Window {
        self.state.window(surface)
    }

    /// Drives the color-management requests `wayland-info` sends: bind an output's image description
    /// and query its information.
    pub fn probe_output_color_management(&mut self) {
        let manager = self.state.color_manager.clone().expect("manager not bound");
        let output = self.state.outputs.keys().next().expect("no output").clone();

        let output_cm = manager.get_output(&output, &self.qh, ());
        let image = output_cm.get_image_description(&self.qh, ());
        let _info = image.get_information(&self.qh, ());
        self.connection.flush().unwrap();
    }

    /// Creates a subsurface of `parent` with a buffer attached and committed, like winewayland does
    /// for Vulkan swapchain presentation. Returns the subsurface's wl_surface.
    pub fn create_committed_subsurface(&mut self, parent: &WlSurface) -> WlSurface {
        let compositor = self.state.compositor.as_ref().unwrap();
        let subcompositor = self
            .state
            .subcompositor
            .as_ref()
            .expect("no wl_subcompositor");
        let spbm = self.state.spbm.as_ref().unwrap();

        let surface = compositor.create_surface(&self.qh, ());
        let _subsurface = subcompositor.get_subsurface(&surface, parent, &self.qh, ());
        let buffer = spbm.create_u32_rgba_buffer(0, 0, 0, u32::MAX, &self.qh, ());
        surface.attach(Some(&buffer), 0, 0);
        surface.commit();
        parent.commit();
        self.connection.flush().unwrap();
        surface
    }

    /// Drives the color-management requests an HDR-aware client (SDL3) sends at startup: create a
    /// surface feedback object and query the preferred image description and its information. The
    /// feedback object is kept alive so later preferred_changed events arrive.
    pub fn probe_surface_preferred(&mut self, surface: &WlSurface) {
        let manager = self.state.color_manager.clone().expect("manager not bound");
        let feedback = manager.get_surface_feedback(surface, &self.qh, ());
        let image = feedback.get_preferred(&self.qh, ());
        let _info = image.get_information(&self.qh, ());
        self.state.surface_feedbacks.push(feedback);
        self.connection.flush().unwrap();
    }

    /// Re-queries the preferred description (without creating a new feedback object) and its info.
    pub fn requery_preferred(&mut self) {
        let feedback = self
            .state
            .surface_feedbacks
            .last()
            .expect("no feedback object");
        let image = feedback.get_preferred(&self.qh, ());
        let _info = image.get_information(&self.qh, ());
        self.connection.flush().unwrap();
    }

    /// Drives the color-management requests an HDR client (mpv gpu-next) sends: build a parametric
    /// image description and attach it to a surface.
    pub fn create_and_attach_hdr_description(
        &mut self,
        surface: &WlSurface,
        tf: TransferFunction,
        primaries: Primaries,
        intent: RenderIntent,
    ) {
        let manager = self.state.color_manager.clone().expect("manager not bound");

        let creator = manager.create_parametric_creator(&self.qh, ());
        creator.set_tf_named(tf);
        creator.set_primaries_named(primaries);
        // min L = 0.005 cd/m² (×10000), max L = 1000 cd/m².
        creator.set_mastering_luminance(50, 1000);
        creator.set_max_cll(1000);
        creator.set_max_fall(400);
        let image = creator.create(&self.qh, ());

        let cm_surface = manager.get_surface(surface, &self.qh, ());
        cm_surface.set_image_description(&image, intent);
        self.connection.flush().unwrap();
    }

    /// Like [`create_and_attach_hdr_description`](Self::create_and_attach_hdr_description), but
    /// additionally sets BT.2020 mastering display primaries — a target color volume that
    /// requires the `extended_target_volume` feature whenever it exceeds the container
    /// primaries.
    pub fn create_and_attach_hdr_description_with_target_volume(
        &mut self,
        surface: &WlSurface,
        tf: TransferFunction,
        primaries: Primaries,
        intent: RenderIntent,
    ) {
        let manager = self.state.color_manager.clone().expect("manager not bound");

        let creator = manager.create_parametric_creator(&self.qh, ());
        creator.set_tf_named(tf);
        creator.set_primaries_named(primaries);
        // BT.2020 primaries and D65 white point, in protocol wire units (xy * 1e6).
        creator.set_mastering_display_primaries(
            708_000, 292_000, 170_000, 797_000, 131_000, 46_000, 312_700, 329_000,
        );
        creator.set_mastering_luminance(50, 1000);
        let image = creator.create(&self.qh, ());

        let cm_surface = manager.get_surface(surface, &self.qh, ());
        cm_surface.set_image_description(&image, intent);
        self.connection.flush().unwrap();
    }

    /// Builds a parametric image description using raw chromaticity coordinates
    /// (`set_primaries`) instead of a named set, and attaches it to a surface.
    pub fn create_and_attach_custom_primaries_description(
        &mut self,
        surface: &WlSurface,
        tf: TransferFunction,
        primaries: [(i32, i32); 4],
        intent: RenderIntent,
    ) {
        let manager = self.state.color_manager.clone().expect("manager not bound");

        let creator = manager.create_parametric_creator(&self.qh, ());
        creator.set_tf_named(tf);
        let [r, g, b, w] = primaries;
        creator.set_primaries(r.0, r.1, g.0, g.1, b.0, b.1, w.0, w.1);
        let image = creator.create(&self.qh, ());

        let cm_surface = manager.get_surface(surface, &self.qh, ());
        cm_surface.set_image_description(&image, intent);
        self.connection.flush().unwrap();
    }

    /// Drives the requests a Windows-scRGB client (winewayland in scRGB mode) sends: create the
    /// pre-defined scRGB image description and attach it to a surface.
    pub fn create_and_attach_scrgb_description(&mut self, surface: &WlSurface) {
        let manager = self.state.color_manager.clone().expect("manager not bound");

        let image = manager.create_windows_scrgb(&self.qh, ());
        let cm_surface = manager.get_surface(surface, &self.qh, ());
        cm_surface.set_image_description(&image, RenderIntent::Perceptual);
        self.connection.flush().unwrap();
    }

    /// Drives the requests winewayland sends for HDR10 (VK_COLOR_SPACE_HDR10_ST2084_EXT)
    /// swapchains: create the pre-defined Windows-BT.2100 image description (v3) and attach it
    /// to a surface.
    pub fn create_and_attach_bt2100_description(&mut self, surface: &WlSurface) {
        let manager = self.state.color_manager.clone().expect("manager not bound");

        let image = manager.create_windows_bt2100(&self.qh, ());
        let cm_surface = manager.get_surface(surface, &self.qh, ());
        cm_surface.set_image_description(&image, RenderIntent::Perceptual);
        self.connection.flush().unwrap();
    }

    pub fn create_layer(
        &mut self,
        output: Option<&WlOutput>,
        layer: zwlr_layer_shell_v1::Layer,
        namespace: &str,
    ) -> &mut LayerSurface {
        self.state.create_layer(output, layer, namespace.to_owned())
    }

    pub fn layer(&mut self, surface: &WlSurface) -> &mut LayerSurface {
        self.state.layer(surface)
    }

    pub fn output(&mut self, name: &str) -> WlOutput {
        self.state
            .outputs
            .iter()
            .find(|(_, v)| *v == name)
            .unwrap()
            .0
            .clone()
    }
}

impl State {
    pub fn create_window(&mut self) -> &mut Window {
        let compositor = self.compositor.as_ref().unwrap();
        let xdg_wm_base = self.xdg_wm_base.as_ref().unwrap();
        let viewporter = self.viewporter.as_ref().unwrap();

        let surface = compositor.create_surface(&self.qh, ());
        let xdg_surface = xdg_wm_base.get_xdg_surface(&surface, &self.qh, ());
        let xdg_toplevel = xdg_surface.get_toplevel(&self.qh, ());
        let viewport = viewporter.get_viewport(&surface, &self.qh, ());

        let window = Window {
            qh: self.qh.clone(),
            spbm: self.spbm.clone().unwrap(),

            surface,
            xdg_surface,
            xdg_toplevel,
            viewport,
            decoration_manager: self.decoration_manager.clone(),
            decoration: None,
            pending_configure: Configure::default(),
            configures_received: Vec::new(),
            close_requested: false,

            configures_looked_at: 0,
        };

        self.windows.push(window);
        self.windows.last_mut().unwrap()
    }

    pub fn window(&mut self, surface: &WlSurface) -> &mut Window {
        self.windows
            .iter_mut()
            .find(|w| w.surface == *surface)
            .unwrap()
    }

    pub fn create_layer(
        &mut self,
        output: Option<&WlOutput>,
        layer: zwlr_layer_shell_v1::Layer,
        namespace: String,
    ) -> &mut LayerSurface {
        let compositor = self.compositor.as_ref().unwrap();
        let layer_shell = self.layer_shell.as_ref().unwrap();
        let viewporter = self.viewporter.as_ref().unwrap();

        let surface = compositor.create_surface(&self.qh, ());
        let layer_surface =
            layer_shell.get_layer_surface(&surface, output, layer, namespace, &self.qh, ());
        let viewport = viewporter.get_viewport(&surface, &self.qh, ());

        let layer_surface = LayerSurface {
            qh: self.qh.clone(),
            spbm: self.spbm.clone().unwrap(),

            surface,
            layer_surface,
            viewport,
            configures_received: Vec::new(),
            close_requested: false,

            configures_looked_at: 0,
        };

        self.layers.push(layer_surface);
        self.layers.last_mut().unwrap()
    }

    pub fn layer(&mut self, surface: &WlSurface) -> &mut LayerSurface {
        self.layers
            .iter_mut()
            .find(|w| w.surface == *surface)
            .unwrap()
    }
}

impl Window {
    pub fn commit(&self) {
        self.surface.commit();
    }

    pub fn ack_last(&self) {
        let serial = self.configures_received.last().unwrap().0;
        self.xdg_surface.ack_configure(serial);
    }

    pub fn ack_last_and_commit(&self) {
        self.ack_last();
        self.commit();
    }

    pub fn attach_new_buffer(&self) {
        let buffer = self.spbm.create_u32_rgba_buffer(0, 0, 0, 0, &self.qh, ());
        self.surface.attach(Some(&buffer), 0, 0);
    }

    pub fn attach_null(&self) {
        self.surface.attach(None, 0, 0);
    }

    pub fn set_size(&self, w: u16, h: u16) {
        self.viewport.set_destination(i32::from(w), i32::from(h));
    }

    pub fn set_fullscreen(&self, output: Option<&WlOutput>) {
        self.xdg_toplevel.set_fullscreen(output);
    }

    pub fn unset_fullscreen(&self) {
        self.xdg_toplevel.unset_fullscreen();
    }

    pub fn set_maximized(&self) {
        self.xdg_toplevel.set_maximized();
    }

    pub fn unset_maximized(&self) {
        self.xdg_toplevel.unset_maximized();
    }

    pub fn set_parent(&self, parent: Option<&XdgToplevel>) {
        self.xdg_toplevel.set_parent(parent);
    }

    pub fn create_decoration(&mut self) {
        let manager = self
            .decoration_manager
            .as_ref()
            .expect("decoration manager is not bound");
        let decoration = manager.get_toplevel_decoration(&self.xdg_toplevel, &self.qh, ());
        assert!(self.decoration.replace(decoration).is_none());
    }

    pub fn destroy_decoration(&mut self) {
        self.decoration.take().unwrap().destroy();
        // Without a decoration object, the surface is client-side decorated.
        self.pending_configure.decoration_mode = None;
    }

    pub fn set_decoration_mode(&self, mode: zxdg_toplevel_decoration_v1::Mode) {
        self.decoration.as_ref().unwrap().set_mode(mode);
    }

    pub fn unset_decoration_mode(&self) {
        self.decoration.as_ref().unwrap().unset_mode();
    }

    pub fn set_title(&self, title: &str) {
        self.xdg_toplevel.set_title(title.to_owned());
    }

    pub fn recent_configures(&mut self) -> impl Iterator<Item = &Configure> {
        let start = self.configures_looked_at;
        self.configures_looked_at = self.configures_received.len();
        self.configures_received[start..].iter().map(|(_, c)| c)
    }

    pub fn format_recent_configures(&mut self) -> String {
        let mut buf = String::new();
        for configure in self.recent_configures() {
            if !buf.is_empty() {
                buf.push('\n');
            }
            write!(buf, "{configure}").unwrap();
        }
        buf
    }
}

impl LayerSurface {
    pub fn commit(&self) {
        self.surface.commit();
    }

    pub fn ack_last(&self) {
        let serial = self.configures_received.last().unwrap().0;
        self.layer_surface.ack_configure(serial);
    }

    pub fn ack_last_and_commit(&self) {
        self.ack_last();
        self.commit();
    }

    pub fn set_configure_props(&self, props: LayerConfigureProps) {
        let LayerConfigureProps {
            size,
            anchor,
            exclusive_zone,
            margin,
            kb_interactivity,
            layer,
            exclusive_edge,
        } = props;

        if let Some(x) = size {
            self.layer_surface.set_size(x.0, x.1);
        }
        if let Some(x) = anchor {
            self.layer_surface.set_anchor(x);
        }
        if let Some(x) = exclusive_zone {
            self.layer_surface.set_exclusive_zone(x);
        }
        if let Some(x) = margin {
            self.layer_surface
                .set_margin(x.top, x.right, x.bottom, x.left);
        }
        if let Some(x) = kb_interactivity {
            self.layer_surface.set_keyboard_interactivity(x);
        }
        if let Some(x) = layer {
            self.layer_surface.set_layer(x);
        }
        if let Some(x) = exclusive_edge {
            self.layer_surface.set_exclusive_edge(x);
        }
    }

    pub fn attach_new_buffer(&self) {
        let buffer = self.spbm.create_u32_rgba_buffer(0, 0, 0, 0, &self.qh, ());
        self.surface.attach(Some(&buffer), 0, 0);
    }

    pub fn attach_null(&self) {
        self.surface.attach(None, 0, 0);
    }

    pub fn set_size(&self, w: u16, h: u16) {
        self.viewport.set_destination(i32::from(w), i32::from(h));
    }

    pub fn recent_configures(&mut self) -> impl Iterator<Item = &LayerConfigure> {
        let start = self.configures_looked_at;
        self.configures_looked_at = self.configures_received.len();
        self.configures_received[start..].iter().map(|(_, c)| c)
    }

    pub fn format_recent_configures(&mut self) -> String {
        let mut buf = String::new();
        for configure in self.recent_configures() {
            if !buf.is_empty() {
                buf.push('\n');
            }
            write!(buf, "{configure}").unwrap();
        }
        buf
    }
}

impl Dispatch<WlCallback, Arc<SyncData>> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WlCallback,
        event: <WlCallback as wayland_client::Proxy>::Event,
        data: &Arc<SyncData>,
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            wl_callback::Event::Done { .. } => data.done.store(true, Ordering::Relaxed),
            _ => unreachable!(),
        }
    }
}

impl Dispatch<WlRegistry, ()> for State {
    fn event(
        state: &mut Self,
        registry: &WlRegistry,
        event: <WlRegistry as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        match event {
            wl_registry::Event::Global {
                name,
                interface,
                version,
            } => {
                if interface == WlCompositor::interface().name {
                    let version = min(version, WlCompositor::interface().version);
                    state.compositor = Some(registry.bind(name, version, qh, ()));
                } else if interface == XdgWmBase::interface().name {
                    let version = min(version, XdgWmBase::interface().version);
                    state.xdg_wm_base = Some(registry.bind(name, version, qh, ()));
                } else if interface == ZwlrLayerShellV1::interface().name {
                    let version = min(version, ZwlrLayerShellV1::interface().version);
                    state.layer_shell = Some(registry.bind(name, version, qh, ()));
                } else if interface == ZwlrVirtualPointerManagerV1::interface().name {
                    let version = min(version, ZwlrVirtualPointerManagerV1::interface().version);
                    state.virtual_pointer_manager = Some(registry.bind(name, version, qh, ()));
                } else if interface == WpSinglePixelBufferManagerV1::interface().name {
                    let version = min(version, WpSinglePixelBufferManagerV1::interface().version);
                    state.spbm = Some(registry.bind(name, version, qh, ()));
                } else if interface == WpViewporter::interface().name {
                    let version = min(version, WpViewporter::interface().version);
                    state.viewporter = Some(registry.bind(name, version, qh, ()));
                } else if interface == WlSubcompositor::interface().name {
                    let version = min(version, WlSubcompositor::interface().version);
                    state.subcompositor = Some(registry.bind(name, version, qh, ()));
                } else if interface == ZxdgDecorationManagerV1::interface().name {
                    let version = min(version, ZxdgDecorationManagerV1::interface().version);
                    state.decoration_manager = Some(registry.bind(name, version, qh, ()));
                } else if interface == WpColorManagerV1::interface().name {
                    let version = min(version, WpColorManagerV1::interface().version);
                    state.color_manager = Some(registry.bind(name, version, qh, ()));
                } else if interface == WlOutput::interface().name {
                    let version = min(version, WlOutput::interface().version);
                    let output = registry.bind(name, version, qh, ());
                    state.outputs.insert(output, String::new());
                }

                let global = Global {
                    name,
                    interface,
                    version,
                };
                state.globals.push(global);
            }
            wl_registry::Event::GlobalRemove { .. } => (),
            _ => unreachable!(),
        }
    }
}

impl Dispatch<WlOutput, ()> for State {
    fn event(
        state: &mut Self,
        output: &WlOutput,
        event: <WlOutput as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            wl_output::Event::Geometry { .. } => (),
            wl_output::Event::Mode { .. } => (),
            wl_output::Event::Done => (),
            wl_output::Event::Scale { .. } => (),
            wl_output::Event::Name { name } => {
                *state.outputs.get_mut(output).unwrap() = name;
            }
            wl_output::Event::Description { .. } => (),
            _ => unreachable!(),
        }
    }
}

impl Dispatch<WlCompositor, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WlCompositor,
        _event: <WlCompositor as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        unreachable!()
    }
}

impl Dispatch<WlSubcompositor, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WlSubcompositor,
        _event: <WlSubcompositor as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        unreachable!()
    }
}

impl Dispatch<WlSubsurface, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WlSubsurface,
        _event: wl_subsurface::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        unreachable!()
    }
}

impl Dispatch<XdgWmBase, ()> for State {
    fn event(
        _state: &mut Self,
        xdg_wm_base: &XdgWmBase,
        event: <XdgWmBase as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            xdg_wm_base::Event::Ping { serial } => {
                xdg_wm_base.pong(serial);
            }
            _ => unreachable!(),
        }
    }
}

impl Dispatch<ZwlrLayerShellV1, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &ZwlrLayerShellV1,
        _event: <ZwlrLayerShellV1 as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        unreachable!()
    }
}

wayland_client::delegate_noop!(State: ZwlrVirtualPointerManagerV1);
wayland_client::delegate_noop!(State: ZwlrVirtualPointerV1);

impl Dispatch<WlSurface, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WlSurface,
        event: <WlSurface as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            wl_surface::Event::Enter { .. } => (),
            wl_surface::Event::Leave { .. } => (),
            wl_surface::Event::PreferredBufferScale { .. } => (),
            wl_surface::Event::PreferredBufferTransform { .. } => (),
            _ => unreachable!(),
        }
    }
}

impl Dispatch<XdgSurface, ()> for State {
    fn event(
        state: &mut Self,
        xdg_surface: &XdgSurface,
        event: <XdgSurface as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            xdg_surface::Event::Configure { serial } => {
                let window = state
                    .windows
                    .iter_mut()
                    .find(|w| w.xdg_surface == *xdg_surface)
                    .unwrap();
                let configure = window.pending_configure.clone();
                window.configures_received.push((serial, configure));
            }
            _ => unreachable!(),
        }
    }
}

impl Dispatch<XdgToplevel, ()> for State {
    fn event(
        state: &mut Self,
        xdg_toplevel: &XdgToplevel,
        event: <XdgToplevel as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        let window = state
            .windows
            .iter_mut()
            .find(|w| w.xdg_toplevel == *xdg_toplevel)
            .unwrap();

        match event {
            xdg_toplevel::Event::Configure {
                width,
                height,
                states,
            } => {
                let configure = &mut window.pending_configure;
                configure.size = (width, height);
                configure.states = states
                    .chunks_exact(4)
                    .flat_map(TryInto::<[u8; 4]>::try_into)
                    .map(u32::from_ne_bytes)
                    .flat_map(xdg_toplevel::State::try_from)
                    .collect();
            }
            xdg_toplevel::Event::Close => {
                window.close_requested = true;
            }
            xdg_toplevel::Event::ConfigureBounds { width, height } => {
                window.pending_configure.bounds = Some((width, height));
            }
            xdg_toplevel::Event::WmCapabilities { .. } => (),
            _ => unreachable!(),
        }
    }
}

impl Dispatch<ZxdgDecorationManagerV1, ()> for State {
    fn event(
        _state: &mut Self,
        _manager: &ZxdgDecorationManagerV1,
        _event: <ZxdgDecorationManagerV1 as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        unreachable!()
    }
}

impl Dispatch<ZxdgToplevelDecorationV1, ()> for State {
    fn event(
        state: &mut Self,
        decoration: &ZxdgToplevelDecorationV1,
        event: <ZxdgToplevelDecorationV1 as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        let window = state
            .windows
            .iter_mut()
            .find(|w| w.decoration.as_ref() == Some(decoration))
            .unwrap();

        match event {
            zxdg_toplevel_decoration_v1::Event::Configure { mode } => {
                window.pending_configure.decoration_mode = mode.into_result().ok();
            }
            _ => unreachable!(),
        }
    }
}

impl Dispatch<ZwlrLayerSurfaceV1, ()> for State {
    fn event(
        state: &mut Self,
        layer_surface: &ZwlrLayerSurfaceV1,
        event: <ZwlrLayerSurfaceV1 as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        let layer_surface = state
            .layers
            .iter_mut()
            .find(|w| w.layer_surface == *layer_surface)
            .unwrap();

        match event {
            zwlr_layer_surface_v1::Event::Configure {
                serial,
                width,
                height,
            } => {
                let configure = LayerConfigure {
                    size: (width, height),
                };
                layer_surface.configures_received.push((serial, configure));
            }
            zwlr_layer_surface_v1::Event::Closed => layer_surface.close_requested = true,
            _ => unreachable!(),
        }
    }
}

impl Dispatch<WlBuffer, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WlBuffer,
        event: <WlBuffer as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            wl_buffer::Event::Release => (),
            _ => unreachable!(),
        }
    }
}

impl Dispatch<WpSinglePixelBufferManagerV1, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WpSinglePixelBufferManagerV1,
        _event: <WpSinglePixelBufferManagerV1 as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        unreachable!()
    }
}

impl Dispatch<WpViewporter, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WpViewporter,
        _event: <WpViewporter as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        unreachable!()
    }
}

impl Dispatch<WpViewport, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WpViewport,
        _event: <WpViewport as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        unreachable!()
    }
}

impl Dispatch<WpColorManagerV1, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WpColorManagerV1,
        _event: <WpColorManagerV1 as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        // supported_intent / supported_feature / supported_tf_named / supported_primaries_named /
        // done — all ignored; binding alone exercises the server's bind handler.
    }
}

impl Dispatch<WpColorManagementOutputV1, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WpColorManagementOutputV1,
        _event: wp_color_management_output_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WpImageDescriptionV1, ()> for State {
    fn event(
        state: &mut Self,
        _proxy: &WpImageDescriptionV1,
        event: wp_image_description_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            // v1 event; sent to clients binding wp_color_manager_v1 version 1.
            wp_image_description_v1::Event::Ready { identity } => {
                state.ready_identities.push(identity);
            }
            // v2+ replacement with a 64-bit identity.
            wp_image_description_v1::Event::Ready2 {
                identity_hi,
                identity_lo,
            } => {
                assert_eq!(identity_hi, 0, "niri's identities fit in 32 bits");
                state.ready_identities.push(identity_lo);
            }
            _ => {}
        }
    }
}

impl Dispatch<WpColorManagementSurfaceFeedbackV1, ()> for State {
    fn event(
        state: &mut Self,
        _proxy: &WpColorManagementSurfaceFeedbackV1,
        event: wp_color_management_surface_feedback_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            wp_color_management_surface_feedback_v1::Event::PreferredChanged { identity } => {
                state.preferred_changed.push(identity);
            }
            wp_color_management_surface_feedback_v1::Event::PreferredChanged2 {
                identity_hi,
                identity_lo,
            } => {
                assert_eq!(identity_hi, 0, "niri's identities fit in 32 bits");
                state.preferred_changed.push(identity_lo);
            }
            _ => {}
        }
    }
}

impl Dispatch<WpImageDescriptionInfoV1, ()> for State {
    fn event(
        state: &mut Self,
        _proxy: &WpImageDescriptionInfoV1,
        event: wp_image_description_info_v1::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        match event {
            wp_image_description_info_v1::Event::TfNamed { tf } => {
                state.info_tf = tf.into_result().ok();
            }
            wp_image_description_info_v1::Event::PrimariesNamed { primaries } => {
                state.info_primaries = primaries.into_result().ok();
            }
            wp_image_description_info_v1::Event::Luminances {
                min_lum,
                max_lum,
                reference_lum,
            } => {
                state.info_luminances = Some((min_lum, max_lum, reference_lum));
            }
            wp_image_description_info_v1::Event::TargetLuminance { min_lum, max_lum } => {
                state.info_target_luminance = Some((min_lum, max_lum));
            }
            _ => {}
        }
    }
}

impl Dispatch<WpImageDescriptionCreatorParamsV1, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WpImageDescriptionCreatorParamsV1,
        _event: <WpImageDescriptionCreatorParamsV1 as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        // No events.
    }
}

impl Dispatch<WpColorManagementSurfaceV1, ()> for State {
    fn event(
        _state: &mut Self,
        _proxy: &WpColorManagementSurfaceV1,
        _event: <WpColorManagementSurfaceV1 as wayland_client::Proxy>::Event,
        _data: &(),
        _conn: &Connection,
        _qhandle: &QueueHandle<Self>,
    ) {
        // No events.
    }
}
