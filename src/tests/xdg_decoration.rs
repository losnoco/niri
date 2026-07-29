use niri_config::Config;
use smithay::reexports::wayland_protocols::xdg::decoration::zv1::client::zxdg_toplevel_decoration_v1::Mode;
use smithay::reexports::wayland_protocols::xdg::decoration::zv1::server::zxdg_toplevel_decoration_v1::Mode as ServerMode;

use super::client::Window;
use super::*;

fn fixture() -> Fixture {
    let mut config = Config::default();
    // The decoration globals are only visible to clients when we prefer no CSD.
    config.prefer_no_csd = true;
    let mut f = Fixture::with_config(config);
    f.add_output(1, (1920, 1080));
    f
}

fn recent_modes(window: &mut Window) -> Vec<Option<Mode>> {
    window
        .recent_configures()
        .map(|c| c.decoration_mode)
        .collect()
}

#[test]
fn global_advertised_at_version_2() {
    let mut f = fixture();
    let id = f.add_client();

    let client = f.client(id);
    let global = client
        .state
        .globals
        .iter()
        .find(|g| g.interface == "zxdg_decoration_manager_v1")
        .expect("zxdg_decoration_manager_v1 global is missing");
    assert_eq!(global.version, 2);
}

#[test]
fn negotiate_before_initial_commit() {
    let mut f = fixture();
    let id = f.add_client();

    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.create_decoration();
    window.commit();
    f.roundtrip(id);

    // The initial configure carries our preferred (server-side) mode.
    let window = f.client(id).window(&surface);
    assert_eq!(recent_modes(window), [Some(Mode::ServerSide)]);

    window.attach_new_buffer();
    window.ack_last_and_commit();
    f.double_roundtrip(id);
}

#[test]
fn client_mode_preference_is_honored() {
    let mut f = fixture();
    let id = f.add_client();

    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.create_decoration();
    window.set_decoration_mode(Mode::ClientSide);
    window.commit();
    f.roundtrip(id);

    // We grant whatever mode the client requests.
    let window = f.client(id).window(&surface);
    assert_eq!(recent_modes(window), [Some(Mode::ClientSide)]);

    window.attach_new_buffer();
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    // Changing the preference after mapping results in a configure with the new mode.
    let window = f.client(id).window(&surface);
    let _ = recent_modes(window);
    window.set_decoration_mode(Mode::ServerSide);
    f.double_roundtrip(id);

    let window = f.client(id).window(&surface);
    let modes = recent_modes(window);
    assert!(!modes.is_empty());
    assert!(
        modes.iter().all(|m| *m == Some(Mode::ServerSide)),
        "{modes:?}"
    );
}

#[test]
fn unset_mode_returns_to_our_preference() {
    let mut f = fixture();
    let id = f.add_client();

    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.create_decoration();
    window.set_decoration_mode(Mode::ClientSide);
    window.commit();
    f.roundtrip(id);

    let window = f.client(id).window(&surface);
    assert_eq!(recent_modes(window), [Some(Mode::ClientSide)]);

    window.attach_new_buffer();
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    // When the client stops preferring a particular mode, we pick our preferred one.
    let window = f.client(id).window(&surface);
    let _ = recent_modes(window);
    window.unset_decoration_mode();
    f.double_roundtrip(id);

    let window = f.client(id).window(&surface);
    let modes = recent_modes(window);
    assert!(!modes.is_empty());
    assert!(
        modes.iter().all(|m| *m == Some(Mode::ServerSide)),
        "{modes:?}"
    );
}

#[test]
fn create_decoration_after_map() {
    let mut f = fixture();
    let id = f.add_client();

    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.commit();
    f.roundtrip(id);

    let window = f.client(id).window(&surface);
    window.attach_new_buffer();
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    // With version 2, the decoration object may be created after the toplevel already has a
    // buffer committed (version 1 made this a protocol error).
    let window = f.client(id).window(&surface);
    let _ = recent_modes(window);
    window.create_decoration();
    f.double_roundtrip(id);

    // The new decoration object receives a configure with our preferred mode.
    let window = f.client(id).window(&surface);
    let modes = recent_modes(window);
    assert!(!modes.is_empty());
    assert!(
        modes.iter().all(|m| *m == Some(Mode::ServerSide)),
        "{modes:?}"
    );
}

#[test]
fn destroy_reverts_to_client_side() {
    let mut f = fixture();
    let id = f.add_client();

    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.create_decoration();
    window.commit();
    f.roundtrip(id);

    let window = f.client(id).window(&surface);
    assert_eq!(recent_modes(window), [Some(Mode::ServerSide)]);

    window.attach_new_buffer();
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    // Destroying the decoration object switches the surface back to client-side decorations.
    let window = f.client(id).window(&surface);
    let _ = recent_modes(window);
    window.destroy_decoration();
    f.double_roundtrip(id);

    // We send a configure so the client can redraw with decorations.
    let window = f.client(id).window(&surface);
    assert!(window.recent_configures().count() >= 1);
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    let mapped = f.niri().layout.windows().next().unwrap().1;
    let mode = mapped
        .toplevel()
        .with_committed_state(|current| current.and_then(|s| s.decoration_mode));
    assert_eq!(mode, Some(ServerMode::ClientSide));
}

#[test]
fn recreate_before_commit_retains_mode() {
    let mut f = fixture();
    let id = f.add_client();

    // Negotiate client-side decorations.
    let window = f.client(id).create_window();
    let surface = window.surface.clone();
    window.create_decoration();
    window.set_decoration_mode(Mode::ClientSide);
    window.commit();
    f.roundtrip(id);

    let window = f.client(id).window(&surface);
    assert_eq!(recent_modes(window), [Some(Mode::ClientSide)]);

    window.attach_new_buffer();
    window.ack_last_and_commit();
    f.double_roundtrip(id);

    // Destroy and immediately recreate the decoration object without a commit in between. The
    // previously negotiated mode must be retained rather than reset to our server-side
    // preference, and the new decoration object must receive a configure of its own.
    let window = f.client(id).window(&surface);
    let _ = recent_modes(window);
    window.destroy_decoration();
    window.create_decoration();
    f.double_roundtrip(id);

    let window = f.client(id).window(&surface);
    let modes = recent_modes(window);
    assert!(!modes.is_empty());
    assert!(
        modes.iter().all(|m| *m == Some(Mode::ClientSide)),
        "{modes:?}"
    );
}
