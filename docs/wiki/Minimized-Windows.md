# Minimized Windows

Minimizing takes a window out of the layout entirely.
A minimized window is not rendered, does not occupy space in any column, and does not receive frame callbacks, so most clients stop drawing while minimized.
It is not closed, and it keeps its size and its floating state, so restoring it puts it back the way it was.

Because niri has no taskbar of its own, a minimized window has no on-screen representation.
The default `Mod+Alt+Shift+H` bind restores the most recently minimized window, so you always have a way back — see [Restoring](#restoring) below.

## Minimizing

```
niri msg action minimize-window
niri msg action minimize-window --id 5
```

The default binds are `Mod+Alt+H` to minimize and `Mod+Alt+Shift+H` to restore, by analogy with Hide on macOS.
(`Mod+H` itself is `focus-column-left`.)

```kdl
binds {
    Mod+Alt+H { minimize-window; }
    Mod+Alt+Shift+H { unminimize-window; }
}
```

`toggle-window-minimized` minimizes a window, or restores it if it is already minimized.
Note that without `--id` it acts on the *focused* window, and a minimized window is never focused, so the no-argument form can only minimize.

## Restoring

There are three ways to get a minimized window back:

- **A taskbar.** Minimized windows are reported over `wlr-foreign-toplevel-management` with the `minimized` state, and activating one restores and focuses it. Waybar's `wlr/taskbar` and similar panels work out of the box.
- **`unminimize-window`.** With `--id`, restores that window. Without an id, restores the most recently minimized window, which makes it usable as a plain "undo minimize" bind.
- **IPC.** `niri msg windows` lists minimized windows with `"is_minimized": true`. Their `workspace_id` is the workspace they will be restored to.

A window is restored to the workspace it was minimized from.
If that workspace is gone by then — for example because its output was disconnected — the window lands on the active workspace instead.

## Windows minimizing themselves

Clients can minimize themselves with `xdg_toplevel.set_minimized`, and niri honors that.

This matters for Wine and Proton games.
A Windows game in exclusive fullscreen leaves fullscreen and minimizes itself whenever it loses focus, which in a scrolling compositor means every time you switch to another window.
The result is that the game vanishes from your layout on every window switch.

Use the [`block-minimize` window rule](./Configuration:-Window-Rules.md#block-minimize) to opt a window out:

```kdl
window-rule {
    match app-id=r#"^borderlands4\.exe$"#

    block-minimize true
}
```

This only ignores the client's own minimize requests; your binds, IPC, and taskbars still work on that window.
