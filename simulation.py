import numpy as np
from numpy.typing import ArrayLike, NDArray
from rich.console import Console
from rich import box
from rich.table import Table

from converter import code2rb, rb2num
import mth
from common import (
    DEFAULT_PEARL_MOTION,
    DEFAULT_PEARL_POSITION,
    DEFAULT_TNT_MOTION_PER_TNT,
)

g = np.float64(0.03)
d = np.float32(0.99)

END_SPAWN_POS = np.array([100.5, 50, 0.5], dtype=np.float64)
_CONSOLE = Console()
_USE_COLOR = (_CONSOLE.color_system is not None) and (not _CONSOLE.no_color)


def _error(text: str) -> None:
    if _USE_COLOR:
        _CONSOLE.print(f"[bold red]ERROR[/bold red]: {text}")
    else:
        _CONSOLE.print(f"ERROR: {text}")


def _info(text: str) -> None:
    if _USE_COLOR:
        _CONSOLE.print(f"[bold cyan]INFO[/bold cyan]: {text}")
    else:
        _CONSOLE.print(f"INFO: {text}")


def simulate_tick(
    pos: NDArray[np.float64], vel: NDArray[np.float64], yaw: np.float32, teleport: int
):
    """
    teleport
    0 -> none
    -1 -> to overworld
    1 -> to end
    """
    vel[1] -= g
    vel *= d
    target_yaw = np.float32(np.arctan2(vel[0], vel[2]) * mth.RADIANS_TO_DEGREES)
    yaw = yaw + 0.2 * mth.wrap_degrees(target_yaw - yaw)

    match teleport:
        case 0:
            pos += vel
        case 1:
            old_yaw = yaw
            yaw = np.float32(90)
            vel = mth.rotate_yaw_vector(vel, old_yaw, yaw)
            pos = END_SPAWN_POS.copy()
        case -1:
            pass
    return pos, vel, yaw


def setup_pearl(pearl_position, pearl_motion, tnt_motion, tnt_count):
    """
    tnt_count:
    (a, b)
    a->(1,1)
    b->(1,-1)
    """

    a = tnt_count[0]
    b = tnt_count[1]
    tnt_num = np.array([a + b, abs(a) + abs(b), a - b], dtype=np.int32)

    pos, vel, yaw = simulate_tick(pearl_position, pearl_motion, np.float32(0), 0)
    vel += tnt_num * tnt_motion
    return pos, vel, yaw


def simulate_pearl(
    tick,
    pearl_position,
    pearl_motion,
    tnt_motion_per_tnt,
    tnt_count,
    to_end_time=0,
    log=False,
) -> NDArray[np.float64]:
    if tick < 0:
        raise ValueError("tick must be a nonnegative integer")
    if to_end_time > tick:
        raise ValueError("to_end_time must be less than tick")
    if to_end_time < 0:
        raise ValueError("to_end_time must be a nonnegative integer")

    pos, vel, yaw = setup_pearl(
        pearl_position, pearl_motion, tnt_motion_per_tnt, tnt_count
    )
    state_rows: list[tuple[str, str, str, str]] = []
    if log:
        state_rows.append(_make_state_row(0, pos, vel, yaw))

    if tick == 0:
        if log:
            _print_state_table(state_rows)
        return pos

    for current_tick in range(1, tick + 1):
        teleport = 1 if current_tick == to_end_time else 0
        pos, vel, yaw = simulate_tick(pos, vel, yaw, teleport)
        if log:
            state_rows.append(_make_state_row(current_tick, pos, vel, yaw))

    if log:
        _print_state_table(state_rows)

    return pos


def _make_state_row(
    tick: int, pos: NDArray[np.float64], vel: NDArray[np.float64], yaw: ArrayLike
) -> tuple[str, str, str, str]:
    return (str(tick), _fmt_vec3(pos), _fmt_vec3(vel), f"{float(yaw):.6f}")


def _print_state_table(rows: list[tuple[str, str, str, str]]) -> None:
    table = Table(
        show_header=True,
        header_style="bold bright_cyan" if _USE_COLOR else "bold",
        box=box.SIMPLE_HEAVY if _USE_COLOR else box.SIMPLE,
        row_styles=(["none", "dim"] if _USE_COLOR else None),
    )
    table.add_column(
        "Tick", justify="left", style=("bright_white" if _USE_COLOR else "")
    )
    table.add_column(
        "Pos", justify="left", style=("bright_green" if _USE_COLOR else "")
    )
    table.add_column(
        "Vel", justify="left", style=("bright_yellow" if _USE_COLOR else "")
    )
    table.add_column(
        "Yaw", justify="left", style=("bright_magenta" if _USE_COLOR else "")
    )
    for row in rows:
        table.add_row(*row)
    _CONSOLE.print(table)


def _fmt_vec3(v: ArrayLike) -> str:
    arr = np.asarray(v, dtype=np.float64)
    return f"({arr[0]:.6f}, {arr[1]:.6f}, {arr[2]:.6f})"


def _read_int(prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            return int(raw)
        except ValueError:
            _error("Invalid int, please retry.")


def _read_num() -> np.ndarray:
    while True:
        if (
            len(
                parts := input("Input num [a:(1,1) b:(1,-1)]: ")
                .strip()
                .replace(",", " ")
                .split()
            )
            != 2
        ):
            _error("Expected 2 integers: a b")
            continue
        try:
            a = int(parts[0])
            b = int(parts[1])
        except ValueError:
            _error("Invalid int format, please retry.")
            continue
        return np.array([a, b], dtype=np.int32)


def _read_rb_to_num() -> np.ndarray:
    while True:
        if (
            len(
                parts := input("Input rb [direction red blue]: ")
                .strip()
                .replace(",", " ")
                .split()
            )
            != 3
        ):
            _error("Expected 3 integers: direction red blue")
            continue
        try:
            direction = int(parts[0])
            r = int(parts[1])
            b = int(parts[2])
            num = rb2num(np.array([direction, r, b], dtype=np.int64))
        except Exception as exc:
            _error(f"Invalid rb: {exc}")
            continue
        return num.astype(np.int32, copy=False)


def _read_code_to_num() -> np.ndarray:
    while True:
        code = input("Input code: ").strip()
        try:
            rb = code2rb(code)
            num = rb2num(rb)
        except Exception as exc:
            _error(f"Invalid code: {exc}")
            continue
        return num.astype(np.int32, copy=False)


if __name__ == "__main__":
    mode_prompt = "Input mode (0 -> num, 1 -> rb, 2 -> code): "
    while True:
        try:
            while (mode := _read_int(mode_prompt)) not in (0, 1, 2):
                _error("Mode must be 0/1/2")

            tick = _read_int("Input tick: ")
            to_end_time = _read_int("Input to_end_time (0 -> no end teleport): ")

            if mode == 0:
                tnt_count = _read_num()
            elif mode == 1:
                tnt_count = _read_rb_to_num()
            else:
                tnt_count = _read_code_to_num()

            pearl_position = DEFAULT_PEARL_POSITION.copy()
            pearl_motion = DEFAULT_PEARL_MOTION.copy()
            tnt_motion_per_tnt = DEFAULT_TNT_MOTION_PER_TNT.copy()

            pos = simulate_pearl(
                tick,
                pearl_position,
                pearl_motion,
                tnt_motion_per_tnt,
                tnt_count,
                to_end_time,
                True,
            )
            if _USE_COLOR:
                _CONSOLE.print(f"[bold green]FINAL_POS[/bold green]={_fmt_vec3(pos)}")
            else:
                _CONSOLE.print(f"FINAL_POS={_fmt_vec3(pos)}")
            _info("Press Ctrl+C to exit, starting next run...")
        except KeyboardInterrupt:
            _CONSOLE.print("\nExit.")
            break
