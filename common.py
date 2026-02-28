import re
from typing import Optional, Union

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

from converter import num2rb, rb2code

DEFAULT_PEARL_POSITION = np.array([0, 252.71360805009243, 0], dtype=np.float64)
DEFAULT_PEARL_MOTION = np.array([0, 0.3827286093776437, 0], dtype=np.float64)
DEFAULT_TNT_MOTION_PER_TNT = np.array(
    [0.6406475114548377, 0.0000041762421424, 0.6406475114548377], dtype=np.float64
)
_CONSOLE = Console()
_USE_COLOR = (_CONSOLE.color_system is not None) and (not _CONSOLE.no_color)


def print_error(text: str) -> None:
    if _USE_COLOR:
        _CONSOLE.print(f"[bold red]ERROR[/bold red]: {text}")
    else:
        _CONSOLE.print(f"ERROR: {text}")


def print_info(text: str) -> None:
    if _USE_COLOR:
        _CONSOLE.print(f"[bold cyan]INFO[/bold cyan]: {text}")
    else:
        _CONSOLE.print(f"INFO: {text}")


def _with_default_hint(prompt: str, default: Optional[Union[float, int]]) -> str:
    if default is None:
        return prompt
    suffix = f" [Enter for default: {default}]"
    return f"{prompt.rstrip()}{suffix}: "


def read_target(default_dimension: Optional[int] = None) -> tuple[float, float, int]:
    while True:
        while True:
            dim_prompt = "Input target dimension (-1 -> nether, 1 -> end)"
            dim_prompt = _with_default_hint(dim_prompt, default_dimension)
            dimension_raw = input(dim_prompt).strip()
            try:
                if dimension_raw == "" and default_dimension is not None:
                    dimension = default_dimension
                else:
                    dimension = int(dimension_raw)
            except ValueError:
                print_error("Invalid dimension, please retry.")
                continue
            if dimension not in (-1, 1):
                print_error("Only -1 (nether) and 1 (end) are supported.")
                continue
            break

        while True:
            parts = [
                p
                for p in re.split(
                    r"[\s,;，；]+", input("Input target pos [x z]: ").strip()
                )
                if p
            ]
            if len(parts) != 2:
                print_error("Expected 2 values: x z")
                continue
            try:
                x = float(parts[0])
                z = float(parts[1])
            except ValueError:
                print_error("Invalid number format, please retry.")
                continue
            return x, z, dimension


def read_nonnegative_float(prompt: str, default: Optional[float] = None) -> float:
    while True:
        raw = input(_with_default_hint(prompt, default)).strip()
        try:
            if raw == "" and default is not None:
                return default
            if (value := float(raw)) >= 0:
                return value
        except ValueError:
            print_error("Invalid float, please retry.")
            continue
        print_error("Value must be >= 0.")


def read_nonnegative_int(prompt: str, default: Optional[int] = None) -> int:
    while True:
        raw = input(_with_default_hint(prompt, default)).strip()
        try:
            if raw == "" and default is not None:
                return default
            if (value := int(raw)) >= 0:
                return value
        except ValueError:
            print_error("Invalid int, please retry.")
            continue
        print_error("Value must be >= 0.")


def sort_results(results: list[dict]) -> None:
    results.sort(
        key=lambda item: (
            item["time"],
            item.get("to_end_time", 0),
            item["distance"],
        )
    )


def print_results(results: list[dict], top_n: int = 20) -> None:
    top_results = results[:top_n]
    print_info(f"matches={len(results)}")
    print_info(f"showing={len(top_results)}")

    has_to_end = any("to_end_time" in item for item in top_results)

    table = Table(
        show_header=True,
        header_style="bold bright_cyan" if _USE_COLOR else "bold",
        box=box.SIMPLE_HEAVY if _USE_COLOR else box.SIMPLE,
        row_styles=(["none", "dim"] if _USE_COLOR else None),
    )
    table.add_column(
        "Time", justify="left", style=("bright_white" if _USE_COLOR else "")
    )
    table.add_column("Code", style=("bright_yellow" if _USE_COLOR else ""))
    if has_to_end:
        table.add_column("End Portal Pos", style=("green" if _USE_COLOR else ""))
    table.add_column("Pos", style=("green" if _USE_COLOR else ""))
    table.add_column("Error", justify="left", style=("magenta" if _USE_COLOR else ""))

    for item in top_results:
        a, b = item["tnt_count"]
        code = rb2code(num2rb(np.array([a, b], dtype=np.int64)))

        pos = f"({item['x']:.6f}, {item['y']:.6f}, {item['z']:.6f})"
        if has_to_end:
            to_end_pos = "-"
            if "to_end_time" in item:
                to_end_prefix = str(item["to_end_time"])
                if "to_end_x" in item and "to_end_y" in item and "to_end_z" in item:
                    to_end_pos = (
                        f"{to_end_prefix} "
                        f"({item['to_end_x']:.6f}, "
                        f"{item['to_end_y']:.6f}, {item['to_end_z']:.6f})"
                    )
                else:
                    to_end_pos = to_end_prefix
            table.add_row(
                str(item["time"]),
                code,
                to_end_pos,
                pos,
                f"{item['distance']:.6f}",
            )
        else:
            table.add_row(
                str(item["time"]),
                code,
                pos,
                f"{item['distance']:.6f}",
            )

    _CONSOLE.print(table)
