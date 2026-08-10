# Minimized Windows

Minimizing takes a window out of the layout entirely.
A minimized window is not rendered, does not occupy space in any column, and does not receive frame callbacks, so most clients stop drawing while minimized.
It is not closed, and it keeps its size and its floating state, so restoring it puts it back the way it was.

By default a small strip of thumbnails in the bottom-left corner of every output shows what is currently minimized; clicking one restores it.
See [The strip](#the-strip) below to move it or turn it off.
The `Mod+Alt+Shift+H` bind also restores the most recently minimized window, so you always have a way back even with the strip disabled.

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

## The strip

Minimized windows are shown as thumbnails in a corner of every output.
Clicking a thumbnail restores that window and focuses it.

```kdl
minimized-windows {
    // off

    // top-left, top-right, bottom-left or bottom-right.
    position "bottom-left"

    // Length of the longer side of a thumbnail, in logical pixels.
    size 96

    // Gap between thumbnails, and between the strip and the output edges.
    gaps 8
}
```

Thumbnails keep their window's aspect ratio and are laid out along a row of `size` height, so windows of different shapes still line up.
If more windows are minimized than fit across the output, the ones that don't fit are simply not shown rather than being shrunk — they are still restorable with `unminimize-window --id` or from a taskbar.

Because minimized windows are suspended, a thumbnail shows the last frame the window drew before it was minimized, not live content.

The strip renders nothing when nothing is minimized, so there is no need for a separate "hide when empty" setting.

## Restoring

There are four ways to get a minimized window back:

- **Clicking its thumbnail** in the strip.
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
