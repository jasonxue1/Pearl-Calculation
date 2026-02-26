import numpy as np
from tqdm import tqdm

import mth
import simulation as sim
from common import (
    DEFAULT_PEARL_MOTION,
    DEFAULT_PEARL_POSITION,
    DEFAULT_TNT_MOTION_PER_TNT,
    print_info,
    print_results,
    read_nonnegative_float,
    read_nonnegative_int,
    read_target,
    sort_results,
)


def calculation(
    target_x: float,
    target_z: float,
    dimension: int,
    max_error: float,
    max_tnt: int,
    max_time: int,
    chunk_size: int = 256,
) -> list[dict]:
    if dimension not in (-1, 1):
        raise ValueError("dimension must be -1 or 1")
    if max_tnt < 0 or max_time < 0 or max_error < 0:
        raise ValueError("max_tnt, max_time, max_error must be nonnegative")

    target_x_sim = target_x
    target_z_sim = target_z
    max_error2_sim = max_error * max_error
    results: list[dict] = []
    base_pos, base_vel, _ = sim.simulate_tick(
        DEFAULT_PEARL_POSITION.copy(),
        DEFAULT_PEARL_MOTION.copy(),
        np.float32(0),
        0,
    )
    if dimension == -1:
        drag = np.float64(sim.d)
        gravity = np.float64(sim.g)
        one_minus_drag = 1.0 - drag
        gravity_coeff = drag * gravity / one_minus_drag

        b_values = np.arange(-max_tnt, max_tnt + 1, dtype=np.int32)
        b_values_f = b_values.astype(np.float64, copy=False)[None, :]
        b_count = int(b_values.size)

        total_combos = (2 * max_tnt + 1) * (2 * max_tnt + 1) * (max_time + 1)
        progress = tqdm(total=total_combos, desc="search", unit="combo")
        try:
            for time in range(max_time + 1):
                if time == 0:
                    sum_vel = np.float64(0)
                else:
                    drag_pow = drag**time
                    sum_vel = drag * (1.0 - drag_pow) / one_minus_drag

                base_x = base_pos[0] + sum_vel * base_vel[0]
                base_z = base_pos[2] + sum_vel * base_vel[2]

                coeff_x_a = sum_vel * DEFAULT_TNT_MOTION_PER_TNT[0]
                coeff_x_b = coeff_x_a
                coeff_z_a = sum_vel * DEFAULT_TNT_MOTION_PER_TNT[2]
                coeff_z_b = -coeff_z_a

                for start_a in range(-max_tnt, max_tnt + 1, chunk_size):
                    end_a = min(max_tnt + 1, start_a + chunk_size)
                    a_values = np.arange(start_a, end_a, dtype=np.int32)
                    a_values_f = a_values.astype(np.float64, copy=False)[:, None]

                    x_grid = base_x + coeff_x_a * a_values_f + coeff_x_b * b_values_f
                    z_grid = base_z + coeff_z_a * a_values_f + coeff_z_b * b_values_f

                    dx = x_grid - target_x_sim
                    dz = z_grid - target_z_sim
                    distance2 = dx * dx + dz * dz

                    matched = np.argwhere(distance2 <= max_error2_sim)
                    for local_a_idx, b_idx in matched:
                        a = int(a_values[local_a_idx])
                        b = int(b_values[b_idx])

                        tnt_y = abs(a) + abs(b)
                        initial_vy = base_vel[1] + tnt_y * DEFAULT_TNT_MOTION_PER_TNT[1]
                        y = (
                            base_pos[1]
                            + initial_vy * sum_vel
                            - gravity_coeff * (time - sum_vel)
                        )

                        dist = float(np.sqrt(distance2[local_a_idx, b_idx]))
                        x_out = float(x_grid[local_a_idx, b_idx])
                        z_out = float(z_grid[local_a_idx, b_idx])

                        results.append(
                            {
                                "tnt_count": (a, b),
                                "time": time,
                                "distance": dist,
                                "x": x_out,
                                "y": float(y),
                                "z": z_out,
                            }
                        )

                    progress.update((end_a - start_a) * b_count)
        finally:
            progress.close()
    else:
        drag = np.float64(sim.d)
        gravity = np.float64(sim.g)
        one_minus_drag = 1.0 - drag
        gravity_coeff = drag * gravity / one_minus_drag
        spawn = sim.END_SPAWN_POS

        n_tnt = 2 * max_tnt + 1
        total_pairs = n_tnt * n_tnt
        total_time_states = max_time * (max_time + 1) // 2
        total_combos = total_pairs * total_time_states
        pair_chunk = min(total_pairs, max(20000, chunk_size * n_tnt))

        post_s1 = np.zeros(max_time + 1, dtype=np.float64)
        for n in range(1, max_time + 1):
            drag_pow_n = drag**n
            post_s1[n] = drag * (1.0 - drag_pow_n) / one_minus_drag

        progress = tqdm(total=total_combos, desc="search", unit="combo")
        try:
            for to_end_time in range(1, max_time + 1):
                drag_pow_t = drag**to_end_time
                gravity_term_t = drag * gravity * (1.0 - drag_pow_t) / one_minus_drag
                yaw_steps = to_end_time
                to_end_s1 = post_s1[to_end_time]

                for start in range(0, total_pairs, pair_chunk):
                    end = min(total_pairs, start + pair_chunk)
                    idx = np.arange(start, end, dtype=np.int64)
                    a = (idx // n_tnt).astype(np.int32) - max_tnt
                    b = (idx % n_tnt).astype(np.int32) - max_tnt

                    a_f = a.astype(np.float64, copy=False)
                    b_f = b.astype(np.float64, copy=False)
                    tnt_x = a_f + b_f
                    tnt_y = np.abs(a_f) + np.abs(b_f)
                    tnt_z = a_f - b_f

                    vel0_x = base_vel[0] + tnt_x * DEFAULT_TNT_MOTION_PER_TNT[0]
                    vel0_y = base_vel[1] + tnt_y * DEFAULT_TNT_MOTION_PER_TNT[1]
                    vel0_z = base_vel[2] + tnt_z * DEFAULT_TNT_MOTION_PER_TNT[2]
                    to_end_x = base_pos[0] + vel0_x * to_end_s1
                    to_end_y = (
                        base_pos[1]
                        + vel0_y * to_end_s1
                        - gravity_coeff * (to_end_time - to_end_s1)
                    )
                    to_end_z = base_pos[2] + vel0_z * to_end_s1

                    target_yaw = np.float32(
                        np.arctan2(vel0_x, vel0_z) * mth.RADIANS_TO_DEGREES
                    )
                    yaw = np.zeros_like(target_yaw, dtype=np.float32)
                    for _ in range(yaw_steps):
                        yaw += np.float32(0.2) * mth.wrap_degrees(target_yaw - yaw)

                    vel_pre_x = vel0_x * drag_pow_t
                    vel_pre_y = vel0_y * drag_pow_t - gravity_term_t
                    vel_pre_z = vel0_z * drag_pow_t

                    rad = np.float32(yaw - np.float32(90.0)) * np.float32(
                        mth.DEGREES_TO_RADIANS
                    )
                    c = mth.cos(rad).astype(np.float64, copy=False)
                    s = mth.sin(rad).astype(np.float64, copy=False)

                    vel_rot_x = vel_pre_x * c + vel_pre_z * s
                    vel_rot_y = vel_pre_y
                    vel_rot_z = vel_pre_z * c - vel_pre_x * s

                    max_end_ticks = max_time - to_end_time
                    for end_ticks in range(0, max_end_ticks + 1):
                        time = to_end_time + end_ticks
                        s1 = post_s1[end_ticks]
                        x = spawn[0] + vel_rot_x * s1
                        z = spawn[2] + vel_rot_z * s1
                        dx = x - target_x_sim
                        dz = z - target_z_sim
                        distance2 = dx * dx + dz * dz

                        matched = np.nonzero(distance2 <= max_error2_sim)[0]
                        if matched.size:
                            y = (
                                spawn[1]
                                + vel_rot_y * s1
                                - gravity_coeff * (end_ticks - s1)
                            )
                            for mi in matched.tolist():
                                results.append(
                                    {
                                        "tnt_count": (int(a[mi]), int(b[mi])),
                                        "time": time,
                                        "to_end_time": to_end_time,
                                        "to_end_x": float(to_end_x[mi]),
                                        "to_end_y": float(to_end_y[mi]),
                                        "to_end_z": float(to_end_z[mi]),
                                        "distance": float(np.sqrt(distance2[mi])),
                                        "x": float(x[mi]),
                                        "y": float(y[mi]),
                                        "z": float(z[mi]),
                                    }
                                )

                        progress.update(end - start)
        finally:
            progress.close()

    sort_results(results)
    return results


def main() -> None:
    while True:
        try:
            x, z, dimension = read_target(default_dimension=-1)
            max_error = read_nonnegative_float("Input max error", default=10.0)
            max_tnt = read_nonnegative_int("Input max TNT count", default=10880)
            max_time = read_nonnegative_int("Input max time", default=20)

            results = calculation(x, z, dimension, max_error, max_tnt, max_time)
            print_results(results, top_n=20)
            print_info("Press Ctrl+C to exit, starting next run...")
        except KeyboardInterrupt:
            print("\nExit.")
            break


if __name__ == "__main__":
    main()
