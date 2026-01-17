import torch
from tqdm import tqdm

import base
import mth


def pearl_calculation(
    max_tnt,
    max_tick,
    max_to_end_time,
    expect_pos,
    max_distance,
    pearl_position,
    pearl_motion,
    tnt_motion,
    GRAVITY,
    DRAG,
    RADIANS_TO_DEGREES,
    DEGREES_TO_RADIANS,
    SIN,
    SCALE,
    COS_OFFSET,
    max_mem_gb=None,
):
    device = expect_pos.device
    dtype = expect_pos.dtype
    if expect_pos.numel() < 2:
        raise ValueError("expect_pos must have at least x and z components")
    expect_xz = expect_pos[:2].to(dtype=dtype, device=device)

    min_tnt0 = -int(max_tnt[0])
    min_tnt1 = -int(max_tnt[1])
    n_tnt0 = int(max_tnt[0]) * 2 + 1
    n_tnt1 = int(max_tnt[1]) * 2 + 1
    total_combos = n_tnt0 * n_tnt1
    max_tick = int(max_tick)
    max_to_end_time = int(max_to_end_time)
    if max_tick <= 0:
        raise ValueError("max_tick must be a positive integer")
    if max_to_end_time < 0:
        raise ValueError("max_to_end_time must be a nonnegative integer")

    to_end_times = torch.arange(
        1, min(max_to_end_time, max_tick - 1) + 1, dtype=torch.int32, device=device
    )
    in_end_time = torch.arange(0, max_tick, dtype=torch.int32, device=device)

    drag = DRAG.to(dtype=dtype)
    one_minus_drag = 1.0 - drag
    gravity_vec = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device) * GRAVITY.to(dtype)
    gravity_coef = gravity_vec * drag / one_minus_drag

    pre_pow = torch.pow(drag, to_end_times.to(dtype))
    yaw_decay = torch.pow(
        torch.tensor(0.8, dtype=torch.float32, device=device), to_end_times
    )

    post_ticks = in_end_time + 1
    post_pow = torch.pow(drag, post_ticks.to(dtype))
    post_s1 = drag * (1.0 - post_pow) / one_minus_drag
    post_s2 = post_ticks.to(dtype) - drag * (1.0 - post_pow) / one_minus_drag

    pos_fixed = torch.tensor([100.5, 50.0, 0.5], dtype=dtype, device=device)

    def estimate_free_bytes():
        if device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            return int(free_bytes * 0.8)
        return None

    results = []
    total_time_pairs = int(
        sum(max_tick - (t.item() + 1) for t in to_end_times)
    )
    total = total_combos * total_time_pairs
    progress = tqdm(total=total, desc="search", unit="combo")
    try:
        with torch.no_grad():
            bytes_per_combo = 1024
            if device.type == "cuda":
                mem_bytes = (
                    int(max_mem_gb * 1024 * 1024 * 1024)
                    if max_mem_gb is not None
                    else estimate_free_bytes()
                )
            else:
                if max_mem_gb is None:
                    raise ValueError("max_mem_gb is required when running on CPU")
                mem_bytes = int(max_mem_gb * 1024 * 1024 * 1024)

            if mem_bytes is not None:
                chunk_size = max(1, min(total_combos, mem_bytes // bytes_per_combo))
            else:
                chunk_size = total_combos

            for p3_idx, p3_val in enumerate(to_end_times):
                d_pow = pre_pow[p3_idx]

                # target_yaw is constant per tnt combo, so yaw approaches it geometrically.
                for start in range(0, total_combos, chunk_size):
                    end = min(total_combos, start + chunk_size)
                    combo_idx = torch.arange(
                        start, end, dtype=torch.int64, device=device
                    )
                    a = combo_idx // n_tnt1 + min_tnt0
                    b = combo_idx % n_tnt1 + min_tnt1
                    a_f = a.to(dtype)
                    b_f = b.to(dtype)

                    tnt_num = torch.stack(
                        [a_f + b_f, torch.abs(a_f) + torch.abs(b_f), a_f - b_f], dim=1
                    )
                    vel0 = tnt_num * tnt_motion + pearl_motion
                    target_yaw = torch.atan2(vel0[:, 0], vel0[:, 2]) * RADIANS_TO_DEGREES
                    target_yaw = target_yaw.to(torch.float32)
                    vel_pre = vel0 * d_pow - gravity_vec * drag * (1.0 - d_pow) / one_minus_drag

                    old_yaw = target_yaw * (1.0 - yaw_decay[p3_idx])
                    yaw_delta = old_yaw - 90.0
                    rad = yaw_delta * DEGREES_TO_RADIANS
                    c = mth.cos(rad, SIN, SCALE, COS_OFFSET)
                    s = mth.sin(rad, SIN, SCALE)

                    x = vel_pre[:, 0]
                    y = vel_pre[:, 1]
                    z = vel_pre[:, 2]
                    vel_rot = torch.stack([x * c + z * s, y, z * c - x * s], dim=-1)

                    for tick in range(int(p3_val.item()) + 1, max_tick + 1):
                        in_idx = tick - int(p3_val.item()) - 1
                        sim_pos = (
                            pos_fixed
                            + vel_rot * post_s1[in_idx]
                            - gravity_coef * post_s2[in_idx]
                        )
                        sim_xz = sim_pos[:, [0, 2]]
                        diff = sim_xz - expect_xz
                        dist2 = (diff * diff).sum(dim=-1)
                        mask = dist2 < (max_distance * max_distance)

                        if mask.any():
                            idxs = mask.nonzero(as_tuple=False).squeeze(1)
                            idxs_cpu = idxs.to("cpu")
                            t0_cpu = a[idxs].to("cpu")
                            t1_cpu = b[idxs].to("cpu")
                            pos_cpu = sim_pos[idxs].detach().to("cpu")
                            dist_cpu = torch.sqrt(dist2[idxs]).to("cpu")

                            for i in range(idxs_cpu.numel()):
                                results.append(
                                    {
                                        "tnt_0": int(t0_cpu[i].item()),
                                        "tnt_1": int(t1_cpu[i].item()),
                                        "to_end_time": int(p3_val.item()),
                                        "tick": int(tick),
                                        "distance": float(dist_cpu[i].item()),
                                        "pos": pos_cpu[i].tolist(),
                                    }
                                )

                        progress.update(int(a.numel()))
    finally:
        progress.close()

    results.sort(key=lambda item: item["tick"])
    return results


if __name__ == "__main__":
    device = base.get_device()
    print("Running on:", device)

    max_tnt = (20000, 20000)  # (-1,-1) (1,-1)
    max_tick = 30
    max_to_end_time = 30
    max_distance = 10
    max_mem_gb = None

    pearl_position = torch.tensor(
        [-68.0, 200.3548026928415, -33.0],
        dtype=torch.float64,
        device=device,
    )

    pearl_motion = torch.tensor(
        [0, -0.340740225070415, 0], dtype=torch.float64, device=device
    )

    tnt_motion_zx_per_tnt = 0.6026793588895138
    tnt_motion_y_per_tnt = 0.004435058914919521

    tnt_motion = torch.tensor(
        [tnt_motion_zx_per_tnt, tnt_motion_y_per_tnt, tnt_motion_zx_per_tnt],
        dtype=torch.float64,
        device=device,
    )

    GRAVITY = torch.tensor(0.03, dtype=torch.float64, device=device)
    DRAG = torch.tensor(0.99, dtype=torch.float32, device=device)

    SIN, SCALE, COS_OFFSET, RADIANS_TO_DEGREES, DEGREES_TO_RADIANS = mth.build_lut(
        device
    )

    expect_pos = torch.tensor([15302, 23221], dtype=torch.float64, device=device)

    results = pearl_calculation(
        max_tnt,
        max_tick,
        max_to_end_time,
        expect_pos,
        max_distance,
        pearl_position,
        pearl_motion,
        tnt_motion,
        GRAVITY,
        DRAG,
        RADIANS_TO_DEGREES,
        DEGREES_TO_RADIANS,
        SIN,
        SCALE,
        COS_OFFSET,
        max_mem_gb,
    )

    print(f"matches={len(results)}")
    for item in results:
        print(item)
