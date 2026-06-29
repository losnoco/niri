//! Minimal, receive-only implementation of `wp-color-management-v1` (staging).
//!
//! This is enough for the Phase 1 HDR use case: a client (e.g. gamescope, mpv with gpu-next) can
//! create a parametric image description (BT.2020 + PQ, or sRGB + gamma 2.2) and attach it to a
//! `wl_surface`. niri stores the parsed description on the surface; the rendering/output layer reads
//! it to decide HDR signalling and (later) color conversion.
//!
//! We advertise the smallest useful subset: the perceptual rendering intent (ignored),
//! the `parametric` feature (plus mastering-metadata features so HDR clients don't error),
//! the gamma22 / sRGB / ST 2084 PQ transfer functions, and the sRGB / BT.2020 primaries. ICC and
//! Windows-scRGB image descriptions are not supported.
//!
//! Modeled on [`crate::protocols::gamma_control`]. Kept independent of any Smithay color-management
//! helper so it can be swapped for upstream later.

use std::sync::Mutex;

use smithay::reexports::wayland_protocols::wp::color_management::v1::server::{
    wp_color_management_output_v1::{self, WpColorManagementOutputV1},
    wp_color_management_surface_feedback_v1::{self, WpColorManagementSurfaceFeedbackV1},
    wp_color_management_surface_v1::{self, WpColorManagementSurfaceV1},
    wp_color_manager_v1::{
        self, Feature, Primaries, RenderIntent, TransferFunction, WpColorManagerV1,
    },
    wp_image_description_creator_params_v1::{self, WpImageDescriptionCreatorParamsV1},
    wp_image_description_info_v1::WpImageDescriptionInfoV1,
    wp_image_description_v1::{self, WpImageDescriptionV1},
};
use smithay::reexports::wayland_server::protocol::wl_output::WlOutput;
use smithay::reexports::wayland_server::protocol::wl_surface::WlSurface;
use smithay::reexports::wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
};
use smithay::wayland::compositor::with_states;

const VERSION: u32 = 1;

/// Transfer characteristic we understand. Anything else is rejected at the protocol level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorTransfer {
    Srgb,
    Gamma22,
    /// SMPTE ST 2084, a.k.a. PQ — the HDR transfer function.
    St2084Pq,
}

/// Color primaries we understand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorPrimaries {
    Srgb,
    Bt2020,
}

impl ColorTransfer {
    fn from_named(tf: TransferFunction) -> Option<Self> {
        match tf {
            TransferFunction::Srgb => Some(Self::Srgb),
            TransferFunction::Gamma22 => Some(Self::Gamma22),
            TransferFunction::St2084Pq => Some(Self::St2084Pq),
            _ => None,
        }
    }

    fn to_named(self) -> TransferFunction {
        match self {
            Self::Srgb => TransferFunction::Srgb,
            Self::Gamma22 => TransferFunction::Gamma22,
            Self::St2084Pq => TransferFunction::St2084Pq,
        }
    }

    /// Whether this transfer function denotes HDR content (PQ).
    pub fn is_hdr(self) -> bool {
        matches!(self, Self::St2084Pq)
    }
}

impl ColorPrimaries {
    fn from_named(primaries: Primaries) -> Option<Self> {
        match primaries {
            Primaries::Srgb => Some(Self::Srgb),
            Primaries::Bt2020 => Some(Self::Bt2020),
            _ => None,
        }
    }

    fn to_named(self) -> Primaries {
        match self {
            Self::Srgb => Primaries::Srgb,
            Self::Bt2020 => Primaries::Bt2020,
        }
    }
}

/// A parsed, immutable image description. This is the niri-internal representation; it is
/// deliberately independent of the wire types so the protocol layer can be replaced later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageDescription {
    pub transfer: ColorTransfer,
    pub primaries: ColorPrimaries,
    /// Maximum content light level, cd/m² (if the client provided it).
    pub max_cll: Option<u16>,
    /// Maximum frame-average light level, cd/m² (if provided).
    pub max_fall: Option<u16>,
    /// Mastering display max luminance, cd/m² (if provided).
    pub mastering_max_luminance: Option<u16>,
}

impl ImageDescription {
    /// sRGB / gamma 2.2 — the default SDR description we hand out for outputs.
    pub const SRGB: Self = Self {
        transfer: ColorTransfer::Srgb,
        primaries: ColorPrimaries::Srgb,
        max_cll: None,
        max_fall: None,
        mastering_max_luminance: None,
    };

    /// Whether this description denotes HDR content (PQ transfer or BT.2020 wide gamut).
    pub fn is_hdr(&self) -> bool {
        self.transfer.is_hdr() || matches!(self.primaries, ColorPrimaries::Bt2020)
    }
}

/// Per-surface storage for the attached image description, kept in the surface's `data_map`.
#[derive(Default)]
struct SurfaceColorData(Mutex<Option<ImageDescription>>);

/// Stores the image description a client attached to `surface` (or clears it with `None`). Returns
/// whether the value actually changed — clients (e.g. mpv) re-attach the same description every
/// frame, and we only want to react/log on real changes.
fn store_surface_image_description(surface: &WlSurface, desc: Option<ImageDescription>) -> bool {
    with_states(surface, |states| {
        states.data_map.insert_if_missing(SurfaceColorData::default);
        let data = states.data_map.get::<SurfaceColorData>().unwrap();
        let mut current = data.0.lock().unwrap();
        if *current == desc {
            false
        } else {
            *current = desc;
            true
        }
    })
}

/// Reads the image description a client attached to `surface`, if any.
pub fn surface_image_description(surface: &WlSurface) -> Option<ImageDescription> {
    with_states(surface, |states| {
        states
            .data_map
            .get::<SurfaceColorData>()
            .and_then(|d| *d.0.lock().unwrap())
    })
}

/// User data for a `wp_image_description_v1`: the parsed image description it represents.
pub struct ImageDescriptionData {
    desc: ImageDescription,
}

/// Accumulated parameters of a `wp_image_description_creator_params_v1`, mutated as the client sets
/// them and validated on `create`.
#[derive(Default)]
pub struct ParamsState {
    transfer: Option<ColorTransfer>,
    primaries: Option<ColorPrimaries>,
    max_cll: Option<u16>,
    max_fall: Option<u16>,
    mastering_max_luminance: Option<u16>,
}

pub struct ColorManagementState {
    next_identity: u32,
}

/// Global data for `wp_color_manager_v1`, carrying the visibility filter. The global is only shown to
/// clients for which the filter returns true — niri uses this to advertise color management only when
/// it can actually honor HDR (TTY backend with an HDR-enabled output).
pub struct ColorManagementGlobalData {
    filter: Box<dyn Fn(&Client) -> bool + Send + Sync>,
}

pub trait ColorManagementHandler {
    fn color_management_state(&mut self) -> &mut ColorManagementState;

    /// Called when a surface's attached image description changes (set or unset). The compositor
    /// should re-evaluate HDR for the surface's output and schedule a redraw.
    fn image_description_changed(&mut self, _surface: &WlSurface) {}

    /// Schedules sending the information events for `info` (describing `desc`), to run *after* the
    /// current request dispatch returns — see [`send_image_description_info`]. This MUST be deferred:
    /// `wp_image_description_info_v1.done` is a destructor event, and destroying the object inside
    /// the very callback that created it corrupts wayland-backend's bookkeeping (it writes the new
    /// object's data after the callback returns, which would then be a use-after-free).
    fn schedule_image_description_info(
        &mut self,
        info: WpImageDescriptionInfoV1,
        desc: ImageDescription,
    );
}

/// Sends the information events describing `desc` on `info`, terminating with the destructor `done`
/// event. Must be called *outside* the request callback that created `info` (e.g. from an event-loop
/// idle), via [`ColorManagementHandler::schedule_image_description_info`].
pub fn send_image_description_info(info: &WpImageDescriptionInfoV1, desc: &ImageDescription) {
    if !info.is_alive() {
        return;
    }
    info.primaries_named(desc.primaries.to_named());
    info.tf_named(desc.transfer.to_named());
    info.done();
}

impl ColorManagementState {
    pub fn new<D, F>(display: &DisplayHandle, filter: F) -> Self
    where
        D: GlobalDispatch<WpColorManagerV1, ColorManagementGlobalData>,
        D: Dispatch<WpColorManagerV1, ()>,
        D: ColorManagementHandler,
        D: 'static,
        F: Fn(&Client) -> bool + Send + Sync + 'static,
    {
        let data = ColorManagementGlobalData {
            filter: Box::new(filter),
        };
        display.create_global::<D, WpColorManagerV1, _>(VERSION, data);
        Self { next_identity: 1 }
    }

    fn next_identity(&mut self) -> u32 {
        let id = self.next_identity;
        self.next_identity = self.next_identity.wrapping_add(1).max(1);
        id
    }
}

fn u32_to_u16(v: u32) -> u16 {
    u16::try_from(v).unwrap_or(u16::MAX)
}

impl<D> GlobalDispatch<WpColorManagerV1, ColorManagementGlobalData, D> for ColorManagementState
where
    D: GlobalDispatch<WpColorManagerV1, ColorManagementGlobalData>
        + Dispatch<WpColorManagerV1, ()>
        + Dispatch<WpColorManagementOutputV1, WlOutput>
        + Dispatch<WpColorManagementSurfaceV1, WlSurface>
        + Dispatch<WpColorManagementSurfaceFeedbackV1, WlSurface>
        + Dispatch<WpImageDescriptionCreatorParamsV1, Mutex<ParamsState>>
        + Dispatch<WpImageDescriptionV1, ImageDescriptionData>
        + Dispatch<WpImageDescriptionInfoV1, ()>
        + ColorManagementHandler
        + 'static,
{
    fn bind(
        _state: &mut D,
        _handle: &DisplayHandle,
        _client: &Client,
        manager: New<WpColorManagerV1>,
        _global_data: &ColorManagementGlobalData,
        data_init: &mut DataInit<'_, D>,
    ) {
        let manager = data_init.init(manager, ());

        // Advertise the minimal supported subset. The order doesn't matter; `done` terminates it.
        manager.supported_intent(RenderIntent::Perceptual);
        manager.supported_feature(Feature::Parametric);
        // So HDR clients can convey mastering metadata without erroring.
        manager.supported_feature(Feature::SetMasteringDisplayPrimaries);
        manager.supported_feature(Feature::SetLuminances);
        manager.supported_tf_named(TransferFunction::Srgb);
        manager.supported_tf_named(TransferFunction::Gamma22);
        manager.supported_tf_named(TransferFunction::St2084Pq);
        manager.supported_primaries_named(Primaries::Srgb);
        manager.supported_primaries_named(Primaries::Bt2020);
        manager.done();
    }

    fn can_view(client: Client, global_data: &ColorManagementGlobalData) -> bool {
        (global_data.filter)(&client)
    }
}

impl<D> Dispatch<WpColorManagerV1, (), D> for ColorManagementState
where
    D: GlobalDispatch<WpColorManagerV1, ColorManagementGlobalData>
        + Dispatch<WpColorManagerV1, ()>
        + Dispatch<WpColorManagementOutputV1, WlOutput>
        + Dispatch<WpColorManagementSurfaceV1, WlSurface>
        + Dispatch<WpColorManagementSurfaceFeedbackV1, WlSurface>
        + Dispatch<WpImageDescriptionCreatorParamsV1, Mutex<ParamsState>>
        + Dispatch<WpImageDescriptionV1, ImageDescriptionData>
        + Dispatch<WpImageDescriptionInfoV1, ()>
        + ColorManagementHandler
        + 'static,
{
    fn request(
        _state: &mut D,
        _client: &Client,
        resource: &WpColorManagerV1,
        request: <WpColorManagerV1 as Resource>::Request,
        _data: &(),
        _dhandle: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        use wp_color_manager_v1::Request;
        match request {
            Request::GetOutput { id, output } => {
                data_init.init(id, output);
            }
            Request::GetSurface { id, surface } => {
                data_init.init(id, surface);
            }
            Request::GetSurfaceFeedback { id, surface } => {
                data_init.init(id, surface);
            }
            Request::CreateParametricCreator { obj } => {
                data_init.init(obj, Mutex::new(ParamsState::default()));
            }
            Request::CreateIccCreator { .. } => {
                resource.post_error(
                    wp_color_manager_v1::Error::UnsupportedFeature,
                    "ICC image descriptions are not supported",
                );
            }
            Request::CreateWindowsScrgb { .. } => {
                resource.post_error(
                    wp_color_manager_v1::Error::UnsupportedFeature,
                    "Windows scRGB image descriptions are not supported",
                );
            }
            Request::Destroy => {}
            _ => {}
        }
    }
}

impl<D> Dispatch<WpColorManagementOutputV1, WlOutput, D> for ColorManagementState
where
    D: GlobalDispatch<WpColorManagerV1, ColorManagementGlobalData>
        + Dispatch<WpColorManagerV1, ()>
        + Dispatch<WpColorManagementOutputV1, WlOutput>
        + Dispatch<WpColorManagementSurfaceV1, WlSurface>
        + Dispatch<WpColorManagementSurfaceFeedbackV1, WlSurface>
        + Dispatch<WpImageDescriptionCreatorParamsV1, Mutex<ParamsState>>
        + Dispatch<WpImageDescriptionV1, ImageDescriptionData>
        + Dispatch<WpImageDescriptionInfoV1, ()>
        + ColorManagementHandler
        + 'static,
{
    fn request(
        state: &mut D,
        _client: &Client,
        _resource: &WpColorManagementOutputV1,
        request: <WpColorManagementOutputV1 as Resource>::Request,
        _data: &WlOutput,
        _dhandle: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        use wp_color_management_output_v1::Request;
        match request {
            Request::GetImageDescription { image_description } => {
                make_ready_description(state, image_description, ImageDescription::SRGB, data_init);
            }
            Request::Destroy => {}
            _ => {}
        }
    }
}

impl<D> Dispatch<WpColorManagementSurfaceV1, WlSurface, D> for ColorManagementState
where
    D: GlobalDispatch<WpColorManagerV1, ColorManagementGlobalData>
        + Dispatch<WpColorManagerV1, ()>
        + Dispatch<WpColorManagementOutputV1, WlOutput>
        + Dispatch<WpColorManagementSurfaceV1, WlSurface>
        + Dispatch<WpColorManagementSurfaceFeedbackV1, WlSurface>
        + Dispatch<WpImageDescriptionCreatorParamsV1, Mutex<ParamsState>>
        + Dispatch<WpImageDescriptionV1, ImageDescriptionData>
        + Dispatch<WpImageDescriptionInfoV1, ()>
        + ColorManagementHandler
        + 'static,
{
    fn request(
        state: &mut D,
        _client: &Client,
        _resource: &WpColorManagementSurfaceV1,
        request: <WpColorManagementSurfaceV1 as Resource>::Request,
        surface: &WlSurface,
        _dhandle: &DisplayHandle,
        _data_init: &mut DataInit<'_, D>,
    ) {
        use wp_color_management_surface_v1::Request;
        match request {
            Request::SetImageDescription {
                image_description,
                render_intent: _,
            } => {
                let desc = image_description
                    .data::<ImageDescriptionData>()
                    .map(|d| d.desc);
                if let Some(desc) = desc {
                    if store_surface_image_description(surface, Some(desc)) {
                        if desc.is_hdr() {
                            debug!(surface = ?surface.id(), ?desc, "client attached an HDR image description");
                        } else {
                            trace!(surface = ?surface.id(), ?desc, "client attached an image description");
                        }
                        state.image_description_changed(surface);
                    }
                }
            }
            Request::UnsetImageDescription => {
                if store_surface_image_description(surface, None) {
                    state.image_description_changed(surface);
                }
            }
            Request::Destroy => {}
            _ => {}
        }
    }
}

impl<D> Dispatch<WpColorManagementSurfaceFeedbackV1, WlSurface, D> for ColorManagementState
where
    D: GlobalDispatch<WpColorManagerV1, ColorManagementGlobalData>
        + Dispatch<WpColorManagerV1, ()>
        + Dispatch<WpColorManagementOutputV1, WlOutput>
        + Dispatch<WpColorManagementSurfaceV1, WlSurface>
        + Dispatch<WpColorManagementSurfaceFeedbackV1, WlSurface>
        + Dispatch<WpImageDescriptionCreatorParamsV1, Mutex<ParamsState>>
        + Dispatch<WpImageDescriptionV1, ImageDescriptionData>
        + Dispatch<WpImageDescriptionInfoV1, ()>
        + ColorManagementHandler
        + 'static,
{
    fn request(
        state: &mut D,
        _client: &Client,
        _resource: &WpColorManagementSurfaceFeedbackV1,
        request: <WpColorManagementSurfaceFeedbackV1 as Resource>::Request,
        _surface: &WlSurface,
        _dhandle: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        use wp_color_management_surface_feedback_v1::Request;
        match request {
            Request::GetPreferred { image_description }
            | Request::GetPreferredParametric { image_description } => {
                make_ready_description(state, image_description, ImageDescription::SRGB, data_init);
            }
            Request::Destroy => {}
            _ => {}
        }
    }
}

impl<D> Dispatch<WpImageDescriptionCreatorParamsV1, Mutex<ParamsState>, D> for ColorManagementState
where
    D: GlobalDispatch<WpColorManagerV1, ColorManagementGlobalData>
        + Dispatch<WpColorManagerV1, ()>
        + Dispatch<WpColorManagementOutputV1, WlOutput>
        + Dispatch<WpColorManagementSurfaceV1, WlSurface>
        + Dispatch<WpColorManagementSurfaceFeedbackV1, WlSurface>
        + Dispatch<WpImageDescriptionCreatorParamsV1, Mutex<ParamsState>>
        + Dispatch<WpImageDescriptionV1, ImageDescriptionData>
        + Dispatch<WpImageDescriptionInfoV1, ()>
        + ColorManagementHandler
        + 'static,
{
    fn request(
        state: &mut D,
        _client: &Client,
        resource: &WpImageDescriptionCreatorParamsV1,
        request: <WpImageDescriptionCreatorParamsV1 as Resource>::Request,
        data: &Mutex<ParamsState>,
        _dhandle: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        use wp_image_description_creator_params_v1::{Error, Request};
        match request {
            Request::SetTfNamed { tf } => {
                let mut params = data.lock().unwrap();
                if params.transfer.is_some() {
                    resource.post_error(Error::AlreadySet, "transfer function already set");
                    return;
                }
                match tf.into_result().ok().and_then(ColorTransfer::from_named) {
                    Some(tf) => params.transfer = Some(tf),
                    None => resource.post_error(Error::InvalidTf, "unsupported transfer function"),
                }
            }
            Request::SetPrimariesNamed { primaries } => {
                let mut params = data.lock().unwrap();
                if params.primaries.is_some() {
                    resource.post_error(Error::AlreadySet, "primaries already set");
                    return;
                }
                match primaries
                    .into_result()
                    .ok()
                    .and_then(ColorPrimaries::from_named)
                {
                    Some(p) => params.primaries = Some(p),
                    None => {
                        resource.post_error(Error::InvalidPrimariesNamed, "unsupported primaries")
                    }
                }
            }
            Request::SetMasteringLuminance {
                min_lum: _,
                max_lum,
            } => {
                data.lock().unwrap().mastering_max_luminance = Some(u32_to_u16(max_lum));
            }
            Request::SetMaxCll { max_cll } => {
                data.lock().unwrap().max_cll = Some(u32_to_u16(max_cll));
            }
            Request::SetMaxFall { max_fall } => {
                data.lock().unwrap().max_fall = Some(u32_to_u16(max_fall));
            }
            // We advertise set_luminances and set_mastering_display_primaries so clients may call
            // them; we just don't use the values yet.
            Request::SetLuminances { .. } | Request::SetMasteringDisplayPrimaries { .. } => {}
            // Not advertised — reject per protocol.
            Request::SetTfPower { .. } => {
                resource.post_error(Error::UnsupportedFeature, "set_tf_power is not supported");
            }
            Request::SetPrimaries { .. } => {
                resource.post_error(Error::UnsupportedFeature, "set_primaries is not supported");
            }
            Request::Create { image_description } => {
                let params = data.lock().unwrap();
                let (Some(transfer), Some(primaries)) = (params.transfer, params.primaries) else {
                    resource.post_error(
                        Error::IncompleteSet,
                        "transfer function and primaries are both required",
                    );
                    return;
                };
                let desc = ImageDescription {
                    transfer,
                    primaries,
                    max_cll: params.max_cll,
                    max_fall: params.max_fall,
                    mastering_max_luminance: params.mastering_max_luminance,
                };
                drop(params);
                make_ready_description(state, image_description, desc, data_init);
            }
            _ => {}
        }
    }
}

impl<D> Dispatch<WpImageDescriptionV1, ImageDescriptionData, D> for ColorManagementState
where
    D: GlobalDispatch<WpColorManagerV1, ColorManagementGlobalData>
        + Dispatch<WpColorManagerV1, ()>
        + Dispatch<WpColorManagementOutputV1, WlOutput>
        + Dispatch<WpColorManagementSurfaceV1, WlSurface>
        + Dispatch<WpColorManagementSurfaceFeedbackV1, WlSurface>
        + Dispatch<WpImageDescriptionCreatorParamsV1, Mutex<ParamsState>>
        + Dispatch<WpImageDescriptionV1, ImageDescriptionData>
        + Dispatch<WpImageDescriptionInfoV1, ()>
        + ColorManagementHandler
        + 'static,
{
    fn request(
        state: &mut D,
        _client: &Client,
        _resource: &WpImageDescriptionV1,
        request: <WpImageDescriptionV1 as Resource>::Request,
        data: &ImageDescriptionData,
        _dhandle: &DisplayHandle,
        data_init: &mut DataInit<'_, D>,
    ) {
        use wp_image_description_v1::Request;
        match request {
            Request::GetInformation { information } => {
                // We are lenient: even for parametric descriptions (where the spec disallows this),
                // we report the named characteristics rather than killing the client. The actual
                // events (ending in the destructor `done`) are sent deferred — see the handler doc.
                let info = data_init.init(information, ());
                state.schedule_image_description_info(info, data.desc);
            }
            Request::Destroy => {}
            _ => {}
        }
    }
}

impl<D> Dispatch<WpImageDescriptionInfoV1, (), D> for ColorManagementState
where
    D: GlobalDispatch<WpColorManagerV1, ColorManagementGlobalData>
        + Dispatch<WpColorManagerV1, ()>
        + Dispatch<WpColorManagementOutputV1, WlOutput>
        + Dispatch<WpColorManagementSurfaceV1, WlSurface>
        + Dispatch<WpColorManagementSurfaceFeedbackV1, WlSurface>
        + Dispatch<WpImageDescriptionCreatorParamsV1, Mutex<ParamsState>>
        + Dispatch<WpImageDescriptionV1, ImageDescriptionData>
        + Dispatch<WpImageDescriptionInfoV1, ()>
        + ColorManagementHandler
        + 'static,
{
    fn request(
        _state: &mut D,
        _client: &Client,
        _resource: &WpImageDescriptionInfoV1,
        _request: <WpImageDescriptionInfoV1 as Resource>::Request,
        _data: &(),
        _dhandle: &DisplayHandle,
        _data_init: &mut DataInit<'_, D>,
    ) {
        // wp_image_description_info_v1 has no requests.
    }
}

/// Initializes a `wp_image_description_v1` carrying `desc` and immediately marks it ready.
fn make_ready_description<D>(
    state: &mut D,
    image_description: New<WpImageDescriptionV1>,
    desc: ImageDescription,
    data_init: &mut DataInit<'_, D>,
) where
    D: GlobalDispatch<WpColorManagerV1, ColorManagementGlobalData>
        + Dispatch<WpColorManagerV1, ()>
        + Dispatch<WpColorManagementOutputV1, WlOutput>
        + Dispatch<WpColorManagementSurfaceV1, WlSurface>
        + Dispatch<WpColorManagementSurfaceFeedbackV1, WlSurface>
        + Dispatch<WpImageDescriptionCreatorParamsV1, Mutex<ParamsState>>
        + Dispatch<WpImageDescriptionV1, ImageDescriptionData>
        + Dispatch<WpImageDescriptionInfoV1, ()>
        + ColorManagementHandler
        + 'static,
{
    let identity = state.color_management_state().next_identity();
    let image = data_init.init(image_description, ImageDescriptionData { desc });
    image.ready(identity);
}

#[macro_export]
macro_rules! delegate_color_management {
    ($(@<$( $lt:tt $( : $clt:tt $(+ $dlt:tt )* )? ),+>)? $ty: ty) => {
        type __WpColorManagerV1 = smithay::reexports::wayland_protocols::wp::color_management::v1::server::wp_color_manager_v1::WpColorManagerV1;
        type __WpCmOutput = smithay::reexports::wayland_protocols::wp::color_management::v1::server::wp_color_management_output_v1::WpColorManagementOutputV1;
        type __WpCmSurface = smithay::reexports::wayland_protocols::wp::color_management::v1::server::wp_color_management_surface_v1::WpColorManagementSurfaceV1;
        type __WpCmSurfaceFeedback = smithay::reexports::wayland_protocols::wp::color_management::v1::server::wp_color_management_surface_feedback_v1::WpColorManagementSurfaceFeedbackV1;
        type __WpCmParams = smithay::reexports::wayland_protocols::wp::color_management::v1::server::wp_image_description_creator_params_v1::WpImageDescriptionCreatorParamsV1;
        type __WpCmImageDesc = smithay::reexports::wayland_protocols::wp::color_management::v1::server::wp_image_description_v1::WpImageDescriptionV1;
        type __WpCmImageDescInfo = smithay::reexports::wayland_protocols::wp::color_management::v1::server::wp_image_description_info_v1::WpImageDescriptionInfoV1;

        smithay::reexports::wayland_server::delegate_global_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            __WpColorManagerV1: $crate::protocols::color_management::ColorManagementGlobalData
        ] => $crate::protocols::color_management::ColorManagementState);

        smithay::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            __WpColorManagerV1: ()
        ] => $crate::protocols::color_management::ColorManagementState);

        smithay::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            __WpCmOutput: smithay::reexports::wayland_server::protocol::wl_output::WlOutput
        ] => $crate::protocols::color_management::ColorManagementState);

        smithay::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            __WpCmSurface: smithay::reexports::wayland_server::protocol::wl_surface::WlSurface
        ] => $crate::protocols::color_management::ColorManagementState);

        smithay::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            __WpCmSurfaceFeedback: smithay::reexports::wayland_server::protocol::wl_surface::WlSurface
        ] => $crate::protocols::color_management::ColorManagementState);

        smithay::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            __WpCmParams: std::sync::Mutex<$crate::protocols::color_management::ParamsState>
        ] => $crate::protocols::color_management::ColorManagementState);

        smithay::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            __WpCmImageDesc: $crate::protocols::color_management::ImageDescriptionData
        ] => $crate::protocols::color_management::ColorManagementState);

        smithay::reexports::wayland_server::delegate_dispatch!($(@< $( $lt $( : $clt $(+ $dlt )* )? ),+ >)? $ty: [
            __WpCmImageDescInfo: ()
        ] => $crate::protocols::color_management::ColorManagementState);
    };
}
