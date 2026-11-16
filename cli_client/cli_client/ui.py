"""Curses backend: colour pairs, drawing a Frame, the key input loop.

All the layout is in `render.py`; this module only maps style names to curses
attributes and curses key codes to the controller's key names.
"""

from __future__ import annotations

import curses
import os
import time
from typing import Optional

from .controller import Client
from .render import STYLES, Frame, render

__all__ = ["run", "translate_key", "STYLE_COLORS"]

POLL_SECONDS = 0.1

# style -> (foreground, background, attributes); -1 = terminal default.
# Cards are bold on a black face so the four suit colours stand out from the felt.
_B, _D, _R = "bold", "dim", "reverse"
STYLE_COLORS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "default": ("default", "default", ()),
    "border": ("default", "default", ()),
    "header": ("default", "default", (_B,)),
    "felt": ("white", "green", ()),
    "pot": ("yellow", "green", (_B,)),
    "lobby": ("white", "green", (_B,)),
    "seat": ("default", "default", ()),
    "seat_me": ("default", "default", (_B,)),
    "seat_acting": ("black", "white", (_B,)),
    "folded": ("default", "default", (_D,)),
    "winner": ("black", "yellow", (_B,)),
    "hearts": ("red", "black", (_B,)),
    "diamonds": ("blue", "black", (_B,)),
    "spades": ("yellow", "black", (_B,)),
    "clubs": ("green", "black", (_B,)),
    "card_back": ("white", "blue", ()),
    "empty_slot": ("white", "green", (_D,)),
    "chip": ("black", "yellow", (_B,)),
    "button": ("black", "white", (_B,)),
    "blind": ("cyan", "default", (_B,)),
    "log": ("default", "default", ()),
    "log_me": ("default", "default", (_B,)),
    "log_board": ("cyan", "default", ()),
    "log_hand": ("magenta", "default", (_B,)),
    "log_win": ("yellow", "default", (_B,)),
    "menu": ("default", "default", ()),
    "menu_title": ("default", "default", (_B,)),
    "menu_selected": ("black", "cyan", (_B,)),
    "prompt": ("yellow", "default", (_B,)),
    "preset": ("cyan", "default", (_B,)),
    "timer": ("yellow", "default", (_B,)),
    "timer_low": ("red", "default", (_B,)),
    "status": ("default", "default", ()),
    "error": ("red", "default", (_B,)),
    "dim": ("default", "default", (_D,)),
    "help": ("white", "blue", ()),
}

# Without colours: attributes only.
_MONO: dict[str, tuple[str, ...]] = {
    "header": (_B,),
    "seat_me": (_B,),
    "seat_acting": (_R,),
    "folded": (_D,),
    "winner": (_R, _B),
    "hearts": (_B,),
    "diamonds": (_B,),
    "spades": (_B,),
    "clubs": (_B,),
    "card_back": (_D,),
    "chip": (_B,),
    "button": (_R,),
    "log_me": (_B,),
    "log_hand": (_B,),
    "menu_title": (_B,),
    "menu_selected": (_R,),
    "prompt": (_B,),
    "error": (_B,),
    "dim": (_D,),
    "help": (_R,),
}


def _attr(names: tuple[str, ...]) -> int:
    value = 0
    for name in names:
        value |= {"bold": curses.A_BOLD, "dim": curses.A_DIM, "reverse": curses.A_REVERSE}[name]
    return value


class Palette:
    def __init__(self) -> None:
        self._attrs: dict[str, int] = {}
        colors = {
            "default": -1,
            "black": curses.COLOR_BLACK,
            "red": curses.COLOR_RED,
            "green": curses.COLOR_GREEN,
            "yellow": curses.COLOR_YELLOW,
            "blue": curses.COLOR_BLUE,
            "magenta": curses.COLOR_MAGENTA,
            "cyan": curses.COLOR_CYAN,
            "white": curses.COLOR_WHITE,
        }
        use_color = False
        if curses.has_colors():
            try:
                curses.start_color()
                try:
                    curses.use_default_colors()
                except curses.error:
                    colors["default"] = curses.COLOR_BLACK
                use_color = curses.COLORS >= 8 and curses.COLOR_PAIRS > len(STYLES)
            except curses.error:
                use_color = False
        for i, style in enumerate(STYLES, start=1):
            if use_color:
                fg, bg, attrs = STYLE_COLORS.get(style, ("default", "default", ()))
                fg_c = colors[fg]
                bg_c = colors[bg]
                if colors["default"] != -1:  # no default colours: white on black
                    fg_c = curses.COLOR_WHITE if fg == "default" else fg_c
                try:
                    curses.init_pair(i, fg_c, bg_c)
                    self._attrs[style] = curses.color_pair(i) | _attr(attrs)
                    continue
                except curses.error:
                    pass
            self._attrs[style] = _attr(_MONO.get(style, ()))

    def attr(self, style: str) -> int:
        return self._attrs.get(style, 0)


def _keymap() -> dict[int, str]:
    return {
        curses.KEY_UP: "up",
        curses.KEY_DOWN: "down",
        curses.KEY_ENTER: "enter",
        10: "enter",
        13: "enter",
        curses.KEY_PPAGE: "pgup",
        curses.KEY_NPAGE: "pgdn",
        curses.KEY_F2: "F2",
        curses.KEY_F3: "F3",
        27: "esc",
        curses.KEY_BACKSPACE: "backspace",
        127: "backspace",
        8: "backspace",
        3: "ctrl-c",
    }


def translate_key(code: int) -> Optional[str]:
    """curses key code -> controller key name (None for keys we ignore)."""
    name = _keymap().get(code)
    if name is not None:
        return name
    if 32 <= code < 127:
        return chr(code)
    return None


def draw(stdscr: "curses.window", frame: Frame, palette: Palette) -> None:
    stdscr.erase()
    for y in range(frame.height):
        for x, text, style in frame.style_runs(y):
            try:
                stdscr.addstr(y, x, text, palette.attr(style))
            except curses.error:
                pass  # the bottom-right cell
    stdscr.noutrefresh()
    curses.doupdate()


def _main(stdscr: "curses.window", client: Client, poll: float) -> int:
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    curses.raw()  # Ctrl-C arrives as a key (3), not as SIGINT mid-update
    stdscr.keypad(True)
    stdscr.timeout(int(poll * 1000))
    palette = Palette()
    keymap = _keymap()
    while not client.done:
        try:
            code = stdscr.getch()
        except KeyboardInterrupt:
            code = 3
        if code == curses.KEY_RESIZE:
            client.dirty = True
        client.drain()
        if code not in (-1, curses.KEY_RESIZE):
            key = keymap.get(code) or (chr(code) if 32 <= code < 127 else None)
            if key is not None:
                client.handle_key(key)
        client.tick(time.monotonic())
        if client.bell:
            client.bell = False
            try:
                curses.beep()
            except curses.error:
                pass
        if client.dirty:
            client.dirty = False
            height, width = stdscr.getmaxyx()
            draw(stdscr, render(client.state, client.ui_view(), width, height), palette)
    return client.exit_code


def run(client: Client, poll: float = POLL_SECONDS) -> int:
    """Run the full-screen UI until the client is done; returns its exit code.
    The terminal is restored on the way out (also on errors)."""
    os.environ.setdefault("ESCDELAY", "25")
    try:
        return curses.wrapper(_main, client, poll)
    except KeyboardInterrupt:
        return client.exit_code
