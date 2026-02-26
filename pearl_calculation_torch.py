import re

import numpy as np
import torch
from tqdm import tqdm

import mth
import pearl_simulation as sim


DEFAULT_PEARL_POSITION = np.array([0, 252.71360805009243, 0], dtype=np.float64)
DEFAULT_PEARL_MOTION = np.array([0, 0.3827286093776437, 0], dtype=np.float64)
DEFAULT_TNT_MOTION_PER_TNT = np.array(
    [0.6406475114548377, 0.0000041762421424, 0.6406475114548377], dtype=np.float64
)

_SIN_LUT_CACHE: dict[str, torch.Tensor] = {}


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _device_dtype(device: torch.device) -> torch.dtype:
    # MPS does not support float64 kernels reliably; use float32 there.
    return torch.float32 if device.type == "mps" else torch.float64


def _wrap_degrees(deg: torch.Tensor) -> torch.Tensor:
    return torch.remainder(deg + 180.0, 360.0) - 180.0


def _get_sin_lut(device: torch.device) -> torch.Tensor:
    key = str(device)
    lut = _SIN_LUT_CACHE.get(key)
    if lut is None:
        lut = torch.from_numpy(mth.SIN).to(device=device)
        _SIN_LUT_CACHE[key] = lut
    return lut


def _torch_sin(v: torch.Tensor, sin_lut: torch.Tensor) -> torch.Tensor:
    idx = (v.to(torch.float32) * float(mth.SCALE)).to(torch.int64)
    idx = torch.remainder(idx, sin_lut.shape[0])
    return sin_lut[idx]


def _torch_cos(v: torch.Tensor, sin_lut: torch.Tensor) -> torch.Tensor:
    idx = (v.to(torch.float32) * float(mth.SCALE) + float(mth.COS_OFFSET)).to(
        torch.int64
    )
    idx = torch.remainder(idx, sin_lut.shape[0])
    return sin_lut[idx]


def _read_target() -> tuple[float, float, int]:
    while True:
        raw = input("Input target x z dimension(-1 nether, 1 end): ").strip()
        parts = [p for p in re.split(r"[\s,;，；]+", raw) if p]
        if len(parts) != 3:
            print("Expected 3 values: x z dimension")
            continue
        try:
            x = float(parts[0])
            z = float(parts[1])
            dimension = int(parts[2])
        except ValueError:
            print("Invalid number format, please retry.")
            continue
        if dimension not in (-1, 1):
            print("Only -1 (nether) and 1 (end) are supported.")
            continue
        return x, z, dimension


def _read_float(prompt: str) -> float:
    while True:
        try:
            value = float(input(prompt).strip())
        except ValueError:
            print("Invalid float, please retry.")
            continue
        if value < 0:
            print("Value must be >= 0.")
            continue
        return value


def _read_int(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt).strip())
        except ValueError:
            print("Invalid int, please retry.")
            continue
        if value < 0:
            print("Value must be >= 0.")
            continue
        return value


def calculation(
    target_x: float,
    target_z: float,
    dimension: int,
    max_error: float,
    max_tnt: int,
    max_time: int,
    device: torch.device | None = None,
    pair_chunk: int = 262_144,
    a_chunk: int = 256,
) -> list[dict]:
    if dimension not in (-1, 1):
        raise ValueError("dimension must be -1 or 1")
    if max_tnt < 0 or max_time < 0 or max_error < 0:
        raise ValueError("max_tnt, max_time, max_error must be nonnegative")

    if device is None:
        device = _get_device()
    dtype = _device_dtype(device)

    target_x_t = torch.tensor(target_x, dtype=dtype, device=device)
    target_z_t = torch.tensor(target_z, dtype=dtype, device=device)
    max_error2_t = torch.tensor(max_error * max_error, dtype=dtype, device=device)

    base_pos_np, base_vel_np, _ = sim.simulate_tick(
        DEFAULT_PEARL_POSITION.copy(),
        DEFAULT_PEARL_MOTION.copy(),
        np.float32(0),
        0,
    )
    base_pos = torch.tensor(base_pos_np, dtype=dtype, device=device)
    base_vel = torch.tensor(base_vel_np, dtype=dtype, device=device)
    tnt_motion = torch.tensor(DEFAULT_TNT_MOTION_PER_TNT, dtype=dtype, device=device)

    drag = torch.tensor(float(sim.d), dtype=dtype, device=device)
    gravity = torch.tensor(float(sim.g), dtype=dtype, device=device)
    one_minus_drag = 1.0 - drag
    gravity_coeff = drag * gravity / one_minus_drag

    time_ids = torch.arange(0, max_time + 1, dtype=dtype, device=device)
    drag_pows = torch.pow(drag, time_ids)
    s1_all = torch.zeros(max_time + 1, dtype=dtype, device=device)
    if max_time > 0:
        s1_all[1:] = drag * (1.0 - drag_pows[1:]) / one_minus_drag

    results: list[dict] = []

    if dimension == -1:
        b_values = torch.arange(-max_tnt, max_tnt + 1, dtype=torch.int32, device=device)
        b_values_f = b_values.to(dtype=dtype).unsqueeze(0)
        b_count = int(b_values.numel())
        total = (2 * max_tnt + 1) * (2 * max_tnt + 1) * (max_time + 1)

        progress = tqdm(total=total, desc="search", unit="combo")
        try:
            for time in range(max_time + 1):
                s1_t = s1_all[time]
                t_float = torch.tensor(float(time), dtype=dtype, device=device)

                base_x = base_pos[0] + s1_t * base_vel[0]
                base_z = base_pos[2] + s1_t * base_vel[2]
                coeff_x = s1_t * tnt_motion[0]
                coeff_z = s1_t * tnt_motion[2]
                y_base = base_pos[1] + base_vel[1] * s1_t - gravity_coeff * (t_float - s1_t)

                for start_a in range(-max_tnt, max_tnt + 1, a_chunk):
                    end_a = min(max_tnt + 1, start_a + a_chunk)
                    a_values = torch.arange(
                        start_a, end_a, dtype=torch.int32, device=device
                    )
                    a_values_f = a_values.to(dtype=dtype).unsqueeze(1)

                    x_grid = base_x + coeff_x * (a_values_f + b_values_f)
                    z_grid = base_z + coeff_z * (a_values_f - b_values_f)

                    dx = x_grid - target_x_t
                    dz = z_grid - target_z_t
                    distance2 = dx * dx + dz * dz

                    matched = torch.nonzero(distance2 <= max_error2_t, as_tuple=False)
                    if matched.numel() > 0:
                        local_a = matched[:, 0]
                        local_b = matched[:, 1]
                        a_hit = a_values[local_a]
                        b_hit = b_values[local_b]
                        x_hit = x_grid[local_a, local_b]
                        z_hit = z_grid[local_a, local_b]
                        dist_hit = torch.sqrt(distance2[local_a, local_b])

                        tnt_y_hit = torch.abs(a_hit) + torch.abs(b_hit)
                        y_hit = y_base + tnt_y_hit.to(dtype=dtype) * tnt_motion[1] * s1_t

                        a_cpu = a_hit.to("cpu").tolist()
                        b_cpu = b_hit.to("cpu").tolist()
                        x_cpu = x_hit.to("cpu").tolist()
                        y_cpu = y_hit.to("cpu").tolist()
                        z_cpu = z_hit.to("cpu").tolist()
                        d_cpu = dist_hit.to("cpu").tolist()

                        for i in range(len(a_cpu)):
                            results.append(
                                {
                                    "tnt_count": (int(a_cpu[i]), int(b_cpu[i])),
                                    "time": time,
                                    "distance": float(d_cpu[i]),
                                    "x": float(x_cpu[i]),
                                    "y": float(y_cpu[i]),
                                    "z": float(z_cpu[i]),
                                }
                            )

                    progress.update((end_a - start_a) * b_count)
        finally:
            progress.close()
    else:
        spawn = torch.tensor(sim.END_SPAWN_POS, dtype=dtype, device=device)
        sin_lut = _get_sin_lut(device)
        n_tnt = 2 * max_tnt + 1
        total_pairs = n_tnt * n_tnt
        total_time_states = max_time * (max_time + 1) // 2
        total = total_pairs * total_time_states
        pair_chunk = max(1, min(pair_chunk, total_pairs))

        radians_to_degrees = torch.tensor(
            float(mth.RADIANS_TO_DEGREES), dtype=dtype, device=device
        )
        degrees_to_radians = torch.tensor(
            float(mth.DEGREES_TO_RADIANS), dtype=torch.float32, device=device
        )

        progress = tqdm(total=total, desc="search", unit="combo")
        try:
            for to_end_time in range(1, max_time + 1):
                drag_pow_t = drag_pows[to_end_time]
                gravity_term_t = drag * gravity * (1.0 - drag_pow_t) / one_minus_drag

                max_end_ticks = max_time - to_end_time
                for start in range(0, total_pairs, pair_chunk):
                    end = min(total_pairs, start + pair_chunk)
                    idx = torch.arange(start, end, dtype=torch.int64, device=device)
                    a = (idx // n_tnt).to(torch.int32) - max_tnt
                    b = (idx % n_tnt).to(torch.int32) - max_tnt

                    a_f = a.to(dtype=dtype)
                    b_f = b.to(dtype=dtype)
                    tnt_x = a_f + b_f
                    tnt_y = torch.abs(a_f) + torch.abs(b_f)
                    tnt_z = a_f - b_f

                    vel0_x = base_vel[0] + tnt_x * tnt_motion[0]
                    vel0_y = base_vel[1] + tnt_y * tnt_motion[1]
                    vel0_z = base_vel[2] + tnt_z * tnt_motion[2]

                    target_yaw = (torch.atan2(vel0_x, vel0_z) * radians_to_degrees).to(
                        torch.float32
                    )
                    yaw = torch.zeros_like(target_yaw)
                    for _ in range(to_end_time):
                        yaw = yaw + 0.2 * _wrap_degrees(target_yaw - yaw)

                    vel_pre_x = vel0_x * drag_pow_t
                    vel_pre_y = vel0_y * drag_pow_t - gravity_term_t
                    vel_pre_z = vel0_z * drag_pow_t

                    rad = (yaw - 90.0) * degrees_to_radians
                    c = _torch_cos(rad, sin_lut).to(dtype=dtype)
                    s = _torch_sin(rad, sin_lut).to(dtype=dtype)
                    vel_rot_x = vel_pre_x * c + vel_pre_z * s
                    vel_rot_y = vel_pre_y
                    vel_rot_z = vel_pre_z * c - vel_pre_x * s

                    for end_ticks in range(0, max_end_ticks + 1):
                        time = to_end_time + end_ticks
                        s1 = s1_all[end_ticks]
                        n_float = torch.tensor(float(end_ticks), dtype=dtype, device=device)

                        x = spawn[0] + vel_rot_x * s1
                        z = spawn[2] + vel_rot_z * s1
                        dx = x - target_x_t
                        dz = z - target_z_t
                        distance2 = dx * dx + dz * dz

                        matched = torch.nonzero(distance2 <= max_error2_t, as_tuple=False)
                        if matched.numel() > 0:
                            matched = matched.flatten()
                            y = spawn[1] + vel_rot_y * s1 - gravity_coeff * (n_float - s1)

                            a_hit = a[matched]
                            b_hit = b[matched]
                            x_hit = x[matched]
                            y_hit = y[matched]
                            z_hit = z[matched]
                            d_hit = torch.sqrt(distance2[matched])

                            a_cpu = a_hit.to("cpu").tolist()
                            b_cpu = b_hit.to("cpu").tolist()
                            x_cpu = x_hit.to("cpu").tolist()
                            y_cpu = y_hit.to("cpu").tolist()
                            z_cpu = z_hit.to("cpu").tolist()
                            d_cpu = d_hit.to("cpu").tolist()

                            for i in range(len(a_cpu)):
                                results.append(
                                    {
                                        "tnt_count": (int(a_cpu[i]), int(b_cpu[i])),
                                        "time": time,
                                        "to_end_time": to_end_time,
                                        "distance": float(d_cpu[i]),
                                        "x": float(x_cpu[i]),
                                        "y": float(y_cpu[i]),
                                        "z": float(z_cpu[i]),
                                    }
                                )

                        progress.update(end - start)
        finally:
            progress.close()

    results.sort(
        key=lambda item: (
            item["time"],
            item.get("to_end_time", 0),
            item["distance"],
        )
    )
    return results


def main() -> None:
    x, z, dimension = _read_target()
    max_error = _read_float("Input max error: ")
    max_tnt = _read_int("Input max TNT count: ")
    max_time = _read_int("Input max time: ")

    device = _get_device()
    print(f"device={device.type}")

    results = calculation(x, z, dimension, max_error, max_tnt, max_time, device=device)
    top_results = results[:20]

    print(f"matches={len(results)}")
    print(f"showing={len(top_results)}")
    for item in top_results:
        a, b = item["tnt_count"]
        to_end_time_str = (
            f" to_end_time={item['to_end_time']}" if "to_end_time" in item else ""
        )
        print(
            f"time={item['time']}{to_end_time_str} "
            f"tnt_count=({a}, {b}) "
            f"pos=({item['x']:.6f}, {item['y']:.6f}, {item['z']:.6f}) "
            f"error={item['distance']:.6f}"
        )


if __name__ == "__main__":
    main()
