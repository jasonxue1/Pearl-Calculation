import numpy as np
import torch
from tqdm import tqdm

import mth
import simulation as sim
from common import (
    DEFAULT_PEARL_MOTION,
    DEFAULT_PEARL_POSITION,
    DEFAULT_TNT_MOTION_PER_TNT,
    print_results,
    print_info,
    read_nonnegative_float,
    read_nonnegative_int,
    read_target,
    sort_results,
)

_SIN_LUT_CACHE: dict[str, torch.Tensor] = {}
_DEFAULT_PAIR_CHUNK = 262_144
_DEFAULT_A_CHUNK = 256
_AUTO_MEMORY_FRACTION = 0.5
_A_CHUNK_ALIGNMENT = 32
_PAIR_CHUNK_ALIGNMENT = 1024
_MAX_AUTO_A_CHUNK = 4096
_MAX_AUTO_PAIR_CHUNK = 1_048_576


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


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return int(torch.empty((), dtype=dtype).element_size())


def _align_chunk(value: int, alignment: int) -> int:
    if value <= 1:
        return 1
    if value < alignment:
        return value
    return max(alignment, (value // alignment) * alignment)


def _get_available_device_memory(device: torch.device) -> int | None:
    try:
        if device.type == "cuda":
            with torch.cuda.device(device):
                free_bytes, _ = torch.cuda.mem_get_info()
            return int(free_bytes)

        if device.type == "mps" and hasattr(torch, "mps"):
            if hasattr(torch.mps, "recommended_max_memory"):
                limit_bytes = int(torch.mps.recommended_max_memory())
                if hasattr(torch.mps, "current_allocated_memory"):
                    used_bytes = int(torch.mps.current_allocated_memory())
                elif hasattr(torch.mps, "driver_allocated_memory"):
                    used_bytes = int(torch.mps.driver_allocated_memory())
                else:
                    used_bytes = 0
                return max(0, limit_bytes - used_bytes)
    except RuntimeError:
        return None

    return None


def _resolve_chunk_sizes(
    *,
    dimension: int,
    max_tnt: int,
    device: torch.device,
    dtype: torch.dtype,
    pair_chunk: int | None,
    a_chunk: int | None,
) -> tuple[int, int]:
    if pair_chunk is not None and pair_chunk <= 0:
        raise ValueError("pair_chunk must be a positive integer")
    if a_chunk is not None and a_chunk <= 0:
        raise ValueError("a_chunk must be a positive integer")

    if pair_chunk is not None and a_chunk is not None:
        return pair_chunk, a_chunk

    available_bytes = _get_available_device_memory(device)
    if available_bytes is None:
        return (
            pair_chunk if pair_chunk is not None else _DEFAULT_PAIR_CHUNK,
            a_chunk if a_chunk is not None else _DEFAULT_A_CHUNK,
        )

    budget_bytes = max(1, int(available_bytes * _AUTO_MEMORY_FRACTION))
    dtype_nbytes = _dtype_nbytes(dtype)
    n_tnt = 2 * max_tnt + 1
    total_pairs = n_tnt * n_tnt

    resolved_a_chunk = a_chunk
    if resolved_a_chunk is None:
        # The overworld path keeps several dense float grids alive at once.
        bytes_per_cell = dtype_nbytes * 8
        bytes_per_row = max(1, n_tnt * bytes_per_cell)
        candidate = budget_bytes // bytes_per_row
        candidate = max(1, min(candidate, n_tnt, _MAX_AUTO_A_CHUNK))
        resolved_a_chunk = _align_chunk(candidate, _A_CHUNK_ALIGNMENT)

    resolved_pair_chunk = pair_chunk
    if resolved_pair_chunk is None:
        # The end path keeps a larger set of 1-D working vectors per TNT pair.
        bytes_per_pair = dtype_nbytes * 24 + 32
        candidate = budget_bytes // max(1, bytes_per_pair)
        candidate = max(1, min(candidate, total_pairs, _MAX_AUTO_PAIR_CHUNK))
        resolved_pair_chunk = _align_chunk(candidate, _PAIR_CHUNK_ALIGNMENT)

    if dimension == -1:
        resolved_pair_chunk = (
            pair_chunk if pair_chunk is not None else _DEFAULT_PAIR_CHUNK
        )
    else:
        resolved_pair_chunk = min(resolved_pair_chunk, total_pairs)

    resolved_a_chunk = min(resolved_a_chunk, n_tnt)
    return resolved_pair_chunk, resolved_a_chunk


def calculation(
    target_x: float,
    target_z: float,
    dimension: int,
    max_error: float,
    max_tnt: int,
    max_time: int,
    device: torch.device | None = None,
    pair_chunk: int | None = None,
    a_chunk: int | None = None,
) -> list[dict]:
    if dimension not in (-1, 1):
        raise ValueError("dimension must be -1 or 1")
    if max_tnt < 0 or max_time < 0 or max_error < 0:
        raise ValueError("max_tnt, max_time, max_error must be nonnegative")

    if device is None:
        device = _get_device()
    dtype = _device_dtype(device)
    pair_chunk, a_chunk = _resolve_chunk_sizes(
        dimension=dimension,
        max_tnt=max_tnt,
        device=device,
        dtype=dtype,
        pair_chunk=pair_chunk,
        a_chunk=a_chunk,
    )

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
                y_base = (
                    base_pos[1] + base_vel[1] * s1_t - gravity_coeff * (t_float - s1_t)
                )

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
                        y_hit = (
                            y_base + tnt_y_hit.to(dtype=dtype) * tnt_motion[1] * s1_t
                        )

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
                to_end_s1 = s1_all[to_end_time]
                to_end_n_float = torch.tensor(
                    float(to_end_time), dtype=dtype, device=device
                )

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
                    to_end_x = base_pos[0] + vel0_x * to_end_s1
                    to_end_y = (
                        base_pos[1]
                        + vel0_y * to_end_s1
                        - gravity_coeff * (to_end_n_float - to_end_s1)
                    )
                    to_end_z = base_pos[2] + vel0_z * to_end_s1

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
                        n_float = torch.tensor(
                            float(end_ticks), dtype=dtype, device=device
                        )

                        x = spawn[0] + vel_rot_x * s1
                        z = spawn[2] + vel_rot_z * s1
                        dx = x - target_x_t
                        dz = z - target_z_t
                        distance2 = dx * dx + dz * dz

                        matched = torch.nonzero(
                            distance2 <= max_error2_t, as_tuple=False
                        )
                        if matched.numel() > 0:
                            matched = matched.flatten()
                            y = (
                                spawn[1]
                                + vel_rot_y * s1
                                - gravity_coeff * (n_float - s1)
                            )

                            a_hit = a[matched]
                            b_hit = b[matched]
                            x_hit = x[matched]
                            y_hit = y[matched]
                            z_hit = z[matched]
                            x_end_hit = to_end_x[matched]
                            y_end_hit = to_end_y[matched]
                            z_end_hit = to_end_z[matched]
                            d_hit = torch.sqrt(distance2[matched])

                            a_cpu = a_hit.to("cpu").tolist()
                            b_cpu = b_hit.to("cpu").tolist()
                            x_cpu = x_hit.to("cpu").tolist()
                            y_cpu = y_hit.to("cpu").tolist()
                            z_cpu = z_hit.to("cpu").tolist()
                            xe_cpu = x_end_hit.to("cpu").tolist()
                            ye_cpu = y_end_hit.to("cpu").tolist()
                            ze_cpu = z_end_hit.to("cpu").tolist()
                            d_cpu = d_hit.to("cpu").tolist()

                            for i in range(len(a_cpu)):
                                results.append(
                                    {
                                        "tnt_count": (int(a_cpu[i]), int(b_cpu[i])),
                                        "time": time,
                                        "to_end_time": to_end_time,
                                        "to_end_x": float(xe_cpu[i]),
                                        "to_end_y": float(ye_cpu[i]),
                                        "to_end_z": float(ze_cpu[i]),
                                        "distance": float(d_cpu[i]),
                                        "x": float(x_cpu[i]),
                                        "y": float(y_cpu[i]),
                                        "z": float(z_cpu[i]),
                                    }
                                )

                        progress.update(end - start)
        finally:
            progress.close()

    sort_results(results)
    return results


def main() -> None:
    device = _get_device()
    print_info(f"device={device.type}")
    while True:
        try:
            x, z, dimension = read_target(default_dimension=-1)
            max_error = read_nonnegative_float("Input max error", default=10.0)
            max_tnt = read_nonnegative_int("Input max TNT count", default=10880)
            max_time = read_nonnegative_int("Input max time", default=20)

            results = calculation(
                x, z, dimension, max_error, max_tnt, max_time, device=device
            )
            print_results(results, top_n=20)
            print_info("Press Ctrl+C to exit, starting next run...")
        except KeyboardInterrupt:
            print("\nExit.")
            break


if __name__ == "__main__":
    main()
