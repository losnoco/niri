//! Tests for the xdg-toplevel suspended state on windows that cannot be visible.
//!
//! While the session is locked or the monitors are powered off, clients are told they are
//! suspended so they get a definitive signal instead of frame-callback-starvation heuristics
//! (Chromium's self-drawn notification toasts get stuck open forever without it).

use smithay::reexports::wayland_protocols::xdg::shell::client::xdg_toplevel;

use super::*;

fn power_off_monitors(f: &mut Fixture) {
    let crate::niri::State { backend, niri } = f.niri_state();
    niri.deactivate_monitors(backend);
}

fn power_on_monitors(f: &mut Fixture) {
    let crate::niri::State { backend, niri } = f.niri_state();
    niri.activate_monitors(backend);
}

#[test]
fn suspended_on_power_off_and_cleared_on_power_on() {
    let mut f = Fixture::new();
    f.add_output(1, (1920, 1080));

    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);
    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    // A mapped, visible window must not be suspended.
    let window = f.client(id).window(&surface);
    for configure in window.recent_configures() {
        assert!(
            !configure.states.contains(&xdg_toplevel::State::Suspended),
            "visible window must not be suspended, got {configure}"
        );
    }

    power_off_monitors(&mut f);
    f.double_roundtrip(id);

    let window = f.client(id).window(&surface);
    let configure = window
        .recent_configures()
        .last()
        .expect("expected a configure after powering off monitors");
    assert!(
        configure.states.contains(&xdg_toplevel::State::Suspended),
        "window must be suspended while monitors are off, got {configure}"
    );

    power_on_monitors(&mut f);
    f.double_roundtrip(id);

    let window = f.client(id).window(&surface);
    let configure = window
        .recent_configures()
        .last()
        .expect("expected a configure after powering monitors back on");
    assert!(
        !configure.states.contains(&xdg_toplevel::State::Suspended),
        "suspended must be cleared when monitors power back on, got {configure}"
    );
}

#[test]
fn window_opening_while_monitors_off_is_suspended_upfront() {
    let mut f = Fixture::new();
    f.add_output(1, (1920, 1080));
    power_off_monitors(&mut f);

    let id = f.add_client();
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.double_roundtrip(id);

    // The initial configure must already carry the suspended state, so clients that probe it
    // once at startup see it.
    let window = f.client(id).window(&surface);
    let configure = window
        .recent_configures()
        .last()
        .expect("expected the initial configure");
    assert!(
        configure.states.contains(&xdg_toplevel::State::Suspended),
        "window opening while monitors are off must be suspended upfront, got {configure}"
    );

    // Map it while the monitors are still off, then power them back on: the window must be
    // unsuspended (this is what lets a stuck client resume and e.g. dismiss its notification
    // toast).
    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    power_on_monitors(&mut f);
    f.double_roundtrip(id);

    let window = f.client(id).window(&surface);
    let configure = window
        .recent_configures()
        .last()
        .expect("expected a configure after powering monitors on");
    assert!(
        !configure.states.contains(&xdg_toplevel::State::Suspended),
        "suspended must be cleared when monitors power on, got {configure}"
    );
}
