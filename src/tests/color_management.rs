//! Reproduction/regression tests for the color-management protocol handlers. These drive the same
//! requests that real clients (wayland-info, mpv gpu-next) send. Because niri builds smithay with
//! `use_system_lib`, a panic in a server dispatch handler unwinds across the C libwayland frame and
//! aborts the process — so any handler panic shows up here as a failing test.

use niri_config::output::Hdr;
use niri_config::{Config, Output};
use smithay::reexports::wayland_protocols::wp::color_management::v1::client::wp_color_manager_v1::{
    Primaries, RenderIntent, TransferFunction,
};

use super::*;

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
