"""Pure rendering: TableState + UIView -> Frame (a grid of styled cells).

No curses here: `ui.py` maps the style names to curses attributes, so the
layout is unit-tested without a terminal.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

from .cards import (
    Segment,
    back_segments,
    card_segments,
    empty_slot_segments,
    fmt_amount,
    segments_text,
)
from .controller import UIView
from .state import STREET_NAMES, SeatState, TableState

__all__ = [
    "STYLES",
    "MIN_WIDTH",
    "MIN_HEIGHT",
    "SEAT_W",
    "SEAT_H",
    "Frame",
    "render",
    "seat_positions",
]

STYLES = (
    "default",
    "border",
    "header",
    "felt",
    "pot",
    "lobby",
    "seat",
    "seat_me",
    "seat_acting",
    "folded",
    "winner",
    "hearts",
    "diamonds",
    "spades",
    "clubs",
    "card_back",
    "empty_slot",
    "chip",
    "button",
    "blind",
    "log",
    "log_me",
    "log_board",
    "log_hand",
    "log_win",
    "menu",
    "menu_title",
    "menu_selected",
    "prompt",
    "preset",
    "timer",
    "timer_low",
    "status",
    "error",
    "dim",
    "help",
)

MIN_WIDTH = 100
MIN_HEIGHT = 30
SEAT_W = 15
SEAT_H = 5
PANEL_H = 5

BOX_UNICODE = {"tl": "┌", "tr": "┐", "bl": "└", "br": "┘", "h": "─", "v": "│",
               "lt": "├", "rt": "┤", "tt": "┬", "bt": "┴"}
BOX_ASCII = {"tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|",
             "lt": "+", "rt": "+", "tt": "+", "bt": "+"}


class Frame:
    """`height` x `width` cells of (char, style)."""

    def __init__(self, width: int, height: int, style: str = "default") -> None:
        self.width = width
        self.height = height
        self.cells = [[(" ", style) for _ in range(width)] for _ in range(height)]

    def put(self, x: int, y: int, text: str, style: Optional[str] = None) -> int:
        """Write `text` at (x, y), clipped to the frame. Style None keeps each
        cell's style (e.g. text on the felt). Returns the x after the text."""
        if 0 <= y < self.height:
            row = self.cells[y]
            for i, ch in enumerate(text):
                cx = x + i
                if 0 <= cx < self.width:
                    row[cx] = (ch, row[cx][1] if style is None else style)
        return x + len(text)

    def put_segments(
        self,
        x: int,
        y: int,
        segments: Iterable[Segment],
        default: Optional[str] = None,
        max_x: Optional[int] = None,
    ) -> int:
        """Write (text, style) segments; style None uses `default`. Stops at
        `max_x` (exclusive)."""
        for text, style in segments:
            if max_x is not None:
                room = max_x - x
                if room <= 0:
                    break
                text = text[:room]
            x = self.put(x, y, text, style if style is not None else default)
        return x

    def fill(self, x: int, y: int, w: int, h: int, style: str, ch: str = " ") -> None:
        for yy in range(y, y + h):
            self.put(x, yy, ch * max(0, w), style)

    def box(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        style: Optional[str] = None,
        title: Optional[str] = None,
        chars: dict = BOX_UNICODE,
        fill: Optional[str] = None,
        title_style: Optional[str] = None,
    ) -> None:
        if w < 2 or h < 2:
            return
        if fill is not None:
            self.fill(x + 1, y + 1, w - 2, h - 2, fill)
        self.put(x, y, chars["tl"] + chars["h"] * (w - 2) + chars["tr"], style)
        self.put(x, y + h - 1, chars["bl"] + chars["h"] * (w - 2) + chars["br"], style)
        for yy in range(y + 1, y + h - 1):
            self.put(x, yy, chars["v"], style)
            self.put(x + w - 1, yy, chars["v"], style)
        if title:
            self.put(x + 1, y, title[: max(0, w - 2)], title_style or style)

    # --- inspection (tests, ui) ------------------------------------------------

    def row_text(self, y: int) -> str:
        return "".join(ch for ch, _ in self.cells[y])

    def text(self) -> str:
        return "\n".join(self.row_text(y) for y in range(self.height))

    def find(self, needle: str) -> Optional[tuple[int, int]]:
        for y in range(self.height):
            x = self.row_text(y).find(needle)
            if x >= 0:
                return x, y
        return None

    def find_all(self, needle: str) -> list[tuple[int, int]]:
        out = []
        for y in range(self.height):
            row = self.row_text(y)
            start = 0
            while True:
                x = row.find(needle, start)
                if x < 0:
                    break
                out.append((x, y))
                start = x + 1
        return out

    def style_at(self, x: int, y: int) -> str:
        return self.cells[y][x][1]

    def styles_of(self, needle: str) -> set[str]:
        """Styles used by the first occurrence of `needle`."""
        pos = self.find(needle)
        if pos is None:
            return set()
        x, y = pos
        return {self.cells[y][x + i][1] for i in range(len(needle))}

    def style_runs(self, y: int) -> list[tuple[int, str, str]]:
        """(x, text, style) runs of equal style in row `y`."""
        runs: list[tuple[int, str, str]] = []
        row = self.cells[y]
        start = 0
        while start < self.width:
            style = row[start][1]
            end = start
            while end < self.width and row[end][1] == style:
                end += 1
            runs.append((start, "".join(ch for ch, _ in row[start:end]), style))
            start = end
        return runs


# --- layout --------------------------------------------------------------------


def seat_positions(
    n: int, my_seat: Optional[int], x0: int, y0: int, x1: int, y1: int
) -> dict[int, tuple[float, float]]:
    """Centre of each seat box on an ellipse inside (x0, y0)-(x1, y1). Seats are
    rotated so my seat is at the bottom centre and go round clockwise."""
    w, h = x1 - x0 + 1, y1 - y0 + 1
    cx, cy = x0 + w / 2.0, y0 + h / 2.0
    rx = max(0.0, (w - SEAT_W) / 2.0)
    ry = max(0.0, (h - SEAT_H) / 2.0)
    base = my_seat if my_seat is not None else 0
    out = {}
    for i in range(n):
        p = (i - base) % n
        angle = math.pi / 2 + 2 * math.pi * p / n  # screen y grows downwards
        out[i] = (cx + rx * math.cos(angle), cy + ry * math.sin(angle))
    return out


def render(state: TableState, ui: UIView, width: int, height: int) -> Frame:
    frame = Frame(width, height)
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        times = "x" if ui.ascii else "×"
        msg = f"Please enlarge the terminal to at least {MIN_WIDTH}{times}{MIN_HEIGHT}"
        frame.put(max(0, (width - len(msg)) // 2), height // 2, msg[:width], "error")
        return frame
    chars = BOX_ASCII if ui.ascii else BOX_UNICODE
    log_w = max(26, min(40, width // 4))
    div = width - log_w - 1  # x of the column divider
    sep = height - PANEL_H - 2  # y of the separator above my panel

    frame.box(0, 0, width, height, "border", chars=chars)
    for y in range(1, height - 1):
        frame.put(div, y, chars["v"], "border")
    frame.put(div, 0, chars["tt"], "border")
    frame.put(div, height - 1, chars["bt"], "border")
    frame.put(1, sep, chars["h"] * (div - 1), "border")
    frame.put(0, sep, chars["lt"], "border")
    frame.put(div, sep, chars["rt"], "border")

    frame.put(2, 0, f" {header_text(state, ui.ascii)} "[: div - 3], "header")
    frame.put(div + 2, 0, " Table log "[: log_w - 2], "header")

    _draw_table(frame, state, ui, 1, 1, div - 1, sep - 1)
    _draw_panel(frame, state, ui, 2, sep + 1, div - 2, height - 2)
    _draw_log(frame, state, ui, div + 2, 1, width - 2, height - 2)
    if ui.show_help:
        _draw_help(frame, ui, 1, 1, div - 1, sep - 1, chars)
    return frame


def header_text(state: TableState, ascii: bool = False) -> str:
    s = state.settings
    dot = " - " if ascii else " · "
    parts = [
        "Omaha Hi-Lo 8/b",
        f"Limit {fmt_amount(s.small_bet_bb)}/{fmt_amount(s.big_bet_bb)} "
        f"(blinds {fmt_amount(s.small_blind_bb)}/{fmt_amount(s.big_blind_bb)})",
    ]
    if state.hand_number:
        parts.append(f"Hand #{state.hand_number}")
    if state.hand_running:
        parts.append(STREET_NAMES.get(state.street, ""))
    parts.append(f"{len(state.seats)}/{s.max_seats} seats")
    return dot.join(p for p in parts if p)


def _draw_table(frame: Frame, state: TableState, ui: UIView, x0: int, y0: int, x1: int, y1: int) -> None:
    w, h = x1 - x0 + 1, y1 - y0 + 1
    cx, cy = x0 + w // 2, y0 + h // 2
    rx, ry = w * 0.31, h * 0.29
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if ((x + 0.5 - cx) / rx) ** 2 + ((y + 0.5 - cy) / ry) ** 2 <= 1.0:
                frame.put(x, y, " ", "felt")

    # Centre: board, pot, side pots or the lobby status.
    board = card_segments(state.board.cards, ui.ascii) + empty_slot_segments(5 - len(state.board.cards))
    _put_centered(frame, cx, cy - 1, board)
    if state.hand_running or state.pot > 0:
        _put_centered(frame, cx, cy + 1, [(f"Pot: {fmt_amount(state.pot)} BB", "pot")])
        if len(state.side_pots) > 1:
            parts = [f"Main {fmt_amount(state.side_pots[0].amount)}"] + [
                f"Side {i} {fmt_amount(p.amount)}" for i, p in enumerate(state.side_pots[1:], 1)
            ]
            _put_centered(frame, cx, cy + 2, [(" · ".join(parts) if not ui.ascii else " - ".join(parts), "pot")])
    if ui.lobby_text:
        _put_centered(frame, cx, cy + 1, [(ui.lobby_text, "lobby")])

    n = len(state.seats)
    if n == 0:
        return
    my_seat = state.my_seat
    positions = seat_positions(n, my_seat, x0, y0, x1, y1)
    acting = state.acting[0] if state.acting is not None else None
    order = [i for i in range(n) if i != my_seat and i != acting]
    order += [i for i in (acting, my_seat) if i is not None and i < n and i not in order]
    seen = set()
    for i in order:
        if i in seen:
            continue
        seen.add(i)
        sx, sy = positions[i]
        # The street bet as a chip between the seat and the centre.
        seat = state.seats[i]
        if seat.street_bet > 0:
            chip = ("o" if ui.ascii else "●") + fmt_amount(seat.street_bet)
            px = round(cx + (sx - cx) * 0.55 - len(chip) / 2)
            py = round(cy + (sy - cy) * 0.55)
            frame.put(px, py, chip, "chip")
    for i in order:
        sx, sy = positions[i]
        bx = int(round(sx - SEAT_W / 2.0))
        by = int(round(sy - SEAT_H / 2.0))
        bx = max(x0, min(bx, x1 - SEAT_W + 1))
        by = max(y0, min(by, y1 - SEAT_H + 1))
        _draw_seat(frame, state, ui, i, bx, by)


def _put_centered(frame: Frame, cx: int, y: int, segments: list[Segment]) -> None:
    width = len(segments_text(segments))
    frame.put_segments(cx - width // 2, y, segments)


def seat_style(state: TableState, index: int) -> str:
    seat = state.seats[index]
    acting = state.acting[0] if state.acting is not None else None
    if acting == index and state.hand_running:
        return "seat_acting"
    if seat.won > 0:
        return "winner"
    if seat.folded or seat.sitting_out:
        return "folded"
    if seat.name == state.my_name:
        return "seat_me"
    return "seat"


def seat_badges(state: TableState, index: int) -> list[str]:
    badges = []
    if state.button == index:
        badges.append("D")
    if state.hand_running or state.hand_number:
        if state.sb_seat == index:
            badges.append("SB")
        if state.bb_seat == index:
            badges.append("BB")
    return badges


def seat_status(state: TableState, seat: SeatState) -> str:
    if seat.won > 0:
        return f"+{fmt_amount(seat.won)}"
    if seat.leaving:
        return "LEAVING"
    if seat.sitting_out:
        return "SITTING OUT"
    if seat.all_in and state.hand_running:
        return "ALL-IN"
    return seat.last_action


def _draw_seat(frame: Frame, state: TableState, ui: UIView, index: int, x: int, y: int) -> None:
    seat = state.seats[index]
    style = seat_style(state, index)
    chars = BOX_ASCII if ui.ascii else BOX_UNICODE
    mine = seat.name == state.my_name
    title = "You" if mine else seat.name
    if mine and len(seat.name) <= SEAT_W - 8:
        title = f"You ({seat.name})"
    frame.box(x, y, SEAT_W, SEAT_H, style, title=title[: SEAT_W - 2], chars=chars, fill=style)
    inner = SEAT_W - 2
    # Row 1: stack and badges.
    frame.put(x + 1, y + 1, f"{fmt_amount(seat.stack)} BB"[:inner], style)
    badges = seat_badges(state, index)
    if badges:
        text = " ".join(badges)
        bx = x + SEAT_W - 1 - len(text)
        for b in badges:
            frame.put(bx, y + 1, b, "button" if b == "D" else "blind")
            bx += len(b) + 1
    # Row 2: status / last action, with the acting countdown.
    status = seat_status(state, seat)
    if state.acting is not None and state.acting[0] == index and state.acting[1] is not None:
        left = max(0, math.ceil(state.acting[1] - ui.now))
        status = f"{status} {left}s".strip()
    frame.put(x + 1, y + 2, status[:inner], style)
    # Row 3: cards.
    cards: list[Segment] = []
    if mine and state.my_cards is not None:
        cards = card_segments(state.my_cards, ui.ascii, brackets=False, sep=" ")
    elif seat.shown is not None:
        cards = card_segments(seat.shown, ui.ascii, brackets=False, sep=" ")
    elif state.hand_running and seat.in_hand and not seat.folded:
        cards = back_segments(4, ui.ascii, brackets=False, sep=" ")
    if cards:
        frame.put_segments(x + 2, y + 3, cards, default=style, max_x=x + SEAT_W - 1)


def _draw_panel(frame: Frame, state: TableState, ui: UIView, x0: int, y0: int, x1: int, y1: int) -> None:
    max_x = x1 + 1
    me = state.me
    # Row 0: my hand, stack and countdown.
    if me is None:
        line: list[Segment] = [("Waiting for a seat…" if state.queued else "Not seated", "dim")]
    else:
        line = [("Your hand: ", "menu_title")]
        if state.my_cards is not None:
            line += card_segments(state.my_cards, ui.ascii)
        else:
            line.append(("—" if not ui.ascii else "-", "dim"))
        line.append((f"   Stack: {fmt_amount(me.stack)} BB", "menu_title"))
    if ui.countdown is not None:
        secs = int(math.ceil(ui.countdown))
        line.append((f"   Time: {secs // 60}:{secs % 60:02d}", "timer_low" if secs <= 10 else "timer"))
    frame.put_segments(x0, y0, line, max_x=max_x)

    # Rows 1-2: the menu (or a key hint).
    if ui.menu:
        x, y = x0, y0 + 1
        if ui.menu_title:
            x = frame.put(x, y, f"{ui.menu_title}: ", "menu_title")
        for i, item in enumerate(ui.menu):
            text = f"[{i + 1}] {item.label}"
            if x + len(text) > max_x and y < y0 + 2:
                x, y = x0, y + 1
            x = frame.put(x, y, text[: max(0, max_x - x)], "menu_selected" if i == ui.selected else "menu")
            x += 3
    else:
        hint = "[x] Check/Fold  [a] Call any  [t] Top up  [l] Leave  [?] Help"
        frame.put(x0, y0 + 1, hint[: max_x - x0], "dim")

    # Row 3: prompt, or the preset and the status.
    if ui.prompt:
        frame.put(x0, y0 + 3, ui.prompt[: max_x - x0], "prompt")
    else:
        x = x0
        if ui.preset:
            x = frame.put(x, y0 + 3, f"Preset: {ui.preset}"[: max_x - x0], "preset") + 3
        if ui.status and x < max_x:
            frame.put(x, y0 + 3, ui.status[: max_x - x], "status")
    # Row 4: errors.
    if ui.error:
        frame.put(x0, y0 + 4, ui.error[: max_x - x0], "error")


def _draw_log(frame: Frame, state: TableState, ui: UIView, x0: int, y0: int, x1: int, y1: int) -> None:
    rows = y1 - y0 + 1
    lines = state.log
    end = max(0, len(lines) - max(0, ui.log_scroll))
    start = max(0, end - rows)
    for k, line in enumerate(lines[start:end]):
        default = "log_me" if line.mine else "log"
        frame.put_segments(x0, y0 + k, line.segments, default=default, max_x=x1 + 1)
    if ui.log_scroll > 0:
        frame.put(x1 - 7, y1, "[more]", "dim")


def _draw_help(frame: Frame, ui: UIView, x0: int, y0: int, x1: int, y1: int, chars: dict) -> None:
    lines = list(ui.help_lines)
    w = min(x1 - x0 + 1, max(len(s) for s in lines) + 4)
    h = min(y1 - y0 + 1, len(lines) + 2)
    x = x0 + (x1 - x0 + 1 - w) // 2
    y = y0 + (y1 - y0 + 1 - h) // 2
    frame.box(x, y, w, h, "help", title=" Keys ", chars=chars, fill="help")
    for i, text in enumerate(lines[: h - 2]):
        frame.put(x + 2, y + 1 + i, text[: w - 4], "help")
