//! Reproduction/regression tests for the color-management protocol handlers. These drive the same
//! requests that real clients (wayland-info, mpv gpu-next) send. Because niri builds smithay with
//! `use_system_lib`, a panic in a server dispatch handler unwinds across the C libwayland frame and
//! aborts the process — so any handler panic shows up here as a failing test.

use niri_config::output::Hdr;
use niri_config::{Config, Output};
use smithay::reexports::wayland_protocols::wp::color_management::v1::client::wp_color_manager_v1::{
    Primaries, RenderIntent, TransferFunction,
};
use wayland_client::protocol::wl_surface::WlSurface;

use super::client::ClientId;
use super::*;
use crate::backend::OutputHdrCaps;

/// A fixture whose config opts an output into HDR, so the (gated) `wp_color_manager_v1` global is
/// advertised. Without an HDR-enabled output, niri does not advertise color management at all.
fn fixture_with_hdr() -> Fixture {
    let mut config = Config::default();
    config.outputs.0.push(Output {
        name: "headless-1".to_owned(),
        hdr: Some(Hdr::default()),
        ..Default::default()
    });
    Fixture::with_config(config)
}

#[test]
fn global_is_advertised_and_bound() {
    let mut f = fixture_with_hdr();
    f.add_output(1, (1920, 1080));

    let id = f.add_client();
    f.double_roundtrip(id);

    assert!(
        f.client(id).state.color_manager.is_some(),
        "wp_color_manager_v1 was not advertised/bound"
    );
}

#[test]
fn global_not_advertised_without_hdr_config() {
    // Default config (no `hdr` on any output) must not advertise color management.
    let mut f = Fixture::new();
    f.add_output(1, (1920, 1080));

    let id = f.add_client();
    f.double_roundtrip(id);

    assert!(
        f.client(id).state.color_manager.is_none(),
        "wp_color_manager_v1 must not be advertised without an HDR-enabled output"
    );
}

#[test]
fn probe_output_image_description_like_wayland_info() {
    let mut f = fixture_with_hdr();
    f.add_output(1, (1920, 1080));

    let id = f.add_client();
    f.double_roundtrip(id);

    // get_output -> get_image_description -> get_information, as wayland-info does.
    f.client(id).probe_output_color_management();
    f.double_roundtrip(id);
}

#[test]
fn create_parametric_hdr_description_like_mpv() {
    let mut f = fixture_with_hdr();
    f.add_output(1, (1920, 1080));

    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);

    // create_parametric_creator -> set BT.2020 + PQ + mastering metadata -> create -> attach to the
    // surface, as mpv --vo=gpu-next does for HDR content.
    f.client(id).create_and_attach_hdr_description(
        &surface,
        TransferFunction::St2084Pq,
        Primaries::Bt2020,
        RenderIntent::Perceptual,
    );
    f.double_roundtrip(id);
}

/// Fixture whose output is `hdr mode="on"` with backend HDR capabilities injected, simulating an
/// HDR-capable monitor on the TTY backend.
fn fixture_with_hdr_mode_on() -> Fixture {
    use niri_config::output::HdrMode;

    let mut config = Config::default();
    config.outputs.0.push(Output {
        name: "headless-1".to_owned(),
        hdr: Some(Hdr {
            mode: HdrMode::On,
            ..Default::default()
        }),
        ..Default::default()
    });
    let mut f = Fixture::with_config(config);
    f.add_output(1, (1920, 1080));

    // The headless backend doesn't probe HDR capabilities; inject them like the TTY backend would.
    f.niri_output(1)
        .user_data()
        .insert_if_missing(|| OutputHdrCaps {
            supported: true,
            max_luminance: 800,
            min_luminance: 100,
            max_frame_avg_luminance: 600,
        });
    f
}

#[test]
fn feedback_preferred_defaults_to_srgb() {
    let mut f = fixture_with_hdr();
    f.add_output(1, (1920, 1080));

    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);

    f.client(id).probe_surface_preferred(&surface);
    f.double_roundtrip(id);

    let client = f.client(id);
    assert_eq!(client.state.info_tf, Some(TransferFunction::Srgb));
    assert_eq!(client.state.info_primaries, Some(Primaries::Srgb));
}

#[test]
fn feedback_preferred_is_pq_with_mode_on() {
    let mut f = fixture_with_hdr_mode_on();

    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);
    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    // An SDL3-style client probes the preferred description once at startup, before going
    // fullscreen. With mode "on" it must see PQ/BT.2020 right away.
    f.client(id).probe_surface_preferred(&surface);
    f.double_roundtrip(id);

    let client = f.client(id);
    assert_eq!(client.state.info_tf, Some(TransferFunction::St2084Pq));
    assert_eq!(client.state.info_primaries, Some(Primaries::Bt2020));
}

#[test]
fn preferred_identities_are_stable() {
    let mut f = fixture_with_hdr_mode_on();

    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);
    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    f.client(id).probe_surface_preferred(&surface);
    f.double_roundtrip(id);
    f.client(id).requery_preferred();
    f.double_roundtrip(id);

    let identities = &f.client(id).state.ready_identities;
    assert!(
        identities.len() >= 2,
        "expected two ready events, got {identities:?}"
    );
    let last_two = &identities[identities.len() - 2..];
    assert_eq!(
        last_two[0], last_two[1],
        "the same preferred description must keep the same identity"
    );
}

#[test]
fn hdr_description_on_subsurface_engages() {
    // winewayland (Proton) attaches the HDR image description to a Vulkan *subsurface* of the
    // toplevel, not the toplevel surface itself. Engagement must search the surface tree.
    let mut f = fixture_with_hdr_mode_on();

    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);
    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.set_fullscreen(None);
    window.ack_last_and_commit();
    f.double_roundtrip(id);
    let window = f.client(id).window(&surface);
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    // Vulkan-style subsurface carrying the HDR description.
    let subsurface = f.client(id).create_committed_subsurface(&surface);
    f.roundtrip(id);
    f.client(id).create_and_attach_hdr_description(
        &subsurface,
        TransferFunction::St2084Pq,
        Primaries::Bt2020,
        RenderIntent::Perceptual,
    );
    f.roundtrip(id);
    // The description is double-buffered; commit the subsurface and sync via the parent.
    let subsurface_clone = subsurface.clone();
    subsurface_clone.commit();
    f.client(id).window(&surface).surface.commit();
    f.double_roundtrip(id);

    let output = f.niri_output(1);
    let desc = f.niri().output_hdr_image_description(&output);
    assert!(
        desc.is_some_and(|d| d.is_hdr()),
        "HDR description on a subsurface must engage HDR, got {desc:?}"
    );
}

/// Sets up a fullscreen window with a committed buffer and returns its surface, the shared
/// preamble of the engagement tests.
fn fullscreen_window(f: &mut Fixture, id: ClientId) -> WlSurface {
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);
    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.set_fullscreen(None);
    window.ack_last_and_commit();
    f.double_roundtrip(id);
    let window = f.client(id).window(&surface);
    window.ack_last_and_commit();
    f.double_roundtrip(id);
    surface
}

#[test]
fn windows_scrgb_engages_hdr() {
    // winewayland in scRGB mode uses create_windows_scrgb rather than a parametric
    // description; it must become ready, attach, and engage HDR like PQ content.
    let mut f = fixture_with_hdr_mode_on();

    let id = f.add_client();
    let surface = fullscreen_window(&mut f, id);

    let ready_before = f.client(id).state.ready_identities.len();
    f.client(id).create_and_attach_scrgb_description(&surface);
    f.roundtrip(id);
    // The description is double-buffered surface state.
    f.client(id).window(&surface).surface.commit();
    f.double_roundtrip(id);

    let ready = &f.client(id).state.ready_identities;
    assert!(
        ready.len() > ready_before,
        "the scRGB image description must deliver ready, got {ready:?}"
    );

    let output = f.niri_output(1);
    let desc = f.niri().output_hdr_image_description(&output);
    assert!(
        desc.is_some_and(|d| d.windows_scrgb),
        "a fullscreen scRGB description must engage HDR with the flag intact, got {desc:?}"
    );
}

#[test]
fn windows_bt2100_engages_hdr() {
    // winewayland maps VK_COLOR_SPACE_HDR10_ST2084_EXT swapchains through the v3
    // create_windows_bt2100 request; the description must become ready, attach, and engage
    // HDR like other PQ content.
    let mut f = fixture_with_hdr_mode_on();

    let id = f.add_client();
    let surface = fullscreen_window(&mut f, id);

    let ready_before = f.client(id).state.ready_identities.len();
    f.client(id).create_and_attach_bt2100_description(&surface);
    f.roundtrip(id);
    f.client(id).window(&surface).surface.commit();
    f.double_roundtrip(id);

    let ready = &f.client(id).state.ready_identities;
    assert!(
        ready.len() > ready_before,
        "the BT.2100 image description must deliver ready, got {ready:?}"
    );

    let output = f.niri_output(1);
    let desc = f.niri().output_hdr_image_description(&output);
    assert!(
        desc.is_some_and(|d| d.windows_bt2100),
        "a fullscreen BT.2100 description must engage HDR with the flag intact, got {desc:?}"
    );
}

#[test]
fn output_description_luminances_satisfy_wine_hdr_check() {
    // winewayland only reports an HDR display to Windows when the output image description's
    // max target luminance exceeds its reference white — both read from the info events. Probe
    // the output like the wine driver does and evaluate its exact condition.
    let mut f = fixture_with_hdr_mode_on();

    let id = f.add_client();
    f.double_roundtrip(id);

    f.client(id).probe_output_color_management();
    f.double_roundtrip(id);

    let client = f.client(id);
    let (_, _, reference_lum) = client
        .state
        .info_luminances
        .expect("the luminances info event must be sent");
    let (_, max_target_lum) = client
        .state
        .info_target_luminance
        .expect("the target_luminance info event must be sent");
    // Reference white 203 cd/m² (default), max target = the injected EDID max of 800 cd/m².
    assert_eq!(reference_lum, 203);
    assert_eq!(max_target_lum, 800);
    assert!(
        max_target_lum > reference_lum,
        "wine's HDR display detection requires max target luminance > reference white"
    );
}

#[test]
fn extended_target_volume_roundtrip() {
    use smithay::wayland::color::management::{Chromaticities, Primaries as ServerPrimaries};

    // A target color volume (BT.2020 mastering primaries) exceeding the container primaries
    // (sRGB) requires the advertised extended_target_volume feature; without it, smithay
    // fails the description gracefully and set_image_description would raise a protocol
    // error. The mastering primaries must round-trip into the engaged description, where the
    // TTY backend forwards them as ST 2086 metadata.
    let mut f = fixture_with_hdr_mode_on();

    let id = f.add_client();
    let surface = fullscreen_window(&mut f, id);

    f.client(id)
        .create_and_attach_hdr_description_with_target_volume(
            &surface,
            TransferFunction::St2084Pq,
            Primaries::Srgb,
            RenderIntent::Perceptual,
        );
    f.roundtrip(id);
    f.client(id).window(&surface).surface.commit();
    f.double_roundtrip(id);

    let output = f.niri_output(1);
    let desc = f
        .niri()
        .output_hdr_image_description(&output)
        .expect("PQ description must engage HDR");
    assert_eq!(
        desc.mastering_primaries,
        Some(Chromaticities::from_named(ServerPrimaries::Bt2020)),
        "client mastering display primaries must be preserved"
    );
    assert_eq!(desc.primaries.named, Some(ServerPrimaries::Srgb));
}

#[test]
fn custom_primaries_description() {
    use smithay::wayland::color::management::Chromaticities;

    // A set_primaries client provides raw chromaticity coordinates instead of a named set;
    // the description must be created (the feature is advertised) and the raw values must be
    // visible server-side, with no named primaries attached.
    let mut f = fixture_with_hdr_mode_on();

    let id = f.add_client();
    let surface = fullscreen_window(&mut f, id);

    // Display P3 coordinates, in protocol wire units (xy * 1e6).
    let p3 = [
        (680_000, 320_000),
        (265_000, 690_000),
        (150_000, 60_000),
        (312_700, 329_000),
    ];
    f.client(id).create_and_attach_custom_primaries_description(
        &surface,
        TransferFunction::St2084Pq,
        p3,
        RenderIntent::Perceptual,
    );
    f.roundtrip(id);
    f.client(id).window(&surface).surface.commit();
    f.double_roundtrip(id);

    let output = f.niri_output(1);
    let desc = f
        .niri()
        .output_hdr_image_description(&output)
        .expect("PQ description must engage HDR");
    assert_eq!(desc.primaries.named, None);
    assert_eq!(
        desc.primaries.values,
        Some(Chromaticities {
            red: p3[0],
            green: p3[1],
            blue: p3[2],
            white: p3[3],
        })
    );
}
