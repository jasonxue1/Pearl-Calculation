import torch
from tqdm import tqdm

import base
import mth
import pearl_simulation


def pearl_calculation(
    max_tnt,
    max_time_before_end,
    max_time_in_end,
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
):
    tnt_0 = torch.arange(
        -max_tnt[0], max_tnt[0] + 1, dtype=torch.int32, device=expect_pos.device
    )
    tnt_1 = torch.arange(
        -max_tnt[1], max_tnt[1] + 1, dtype=torch.int32, device=expect_pos.device
    )
    before_end_time = torch.arange(
        1, max_time_before_end + 1, dtype=torch.int32, device=expect_pos.device
    )
    in_end_time = torch.arange(
        0, max_time_in_end + 1, dtype=torch.int32, device=expect_pos.device
    )

    results = []
    total = (
        int(tnt_0.numel())
        * int(tnt_1.numel())
        * int(before_end_time.numel())
        * int(in_end_time.numel())
    )
    progress = tqdm(total=total, desc="search", unit="combo")
    try:
        with torch.no_grad():
            for p1 in tnt_0.tolist():
                for p2 in tnt_1.tolist():
                    tnt_count = torch.tensor(
                        [p1, p2], dtype=torch.float64, device=expect_pos.device
                    )
                    for p3 in before_end_time.tolist():
                        for p4 in in_end_time.tolist():
                            tick = int(p3 + p4 + 1)
                            to_end_time = int(p3)
                            sim_pos = pearl_simulation.simulate_pearl(
                                tick,
                                to_end_time,
                                False,
                                pearl_position,
                                pearl_motion,
                                tnt_motion,
                                tnt_count,
                                GRAVITY,
                                DRAG,
                                RADIANS_TO_DEGREES,
                                DEGREES_TO_RADIANS,
                                SIN,
                                SCALE,
                                COS_OFFSET,
                            )

                            distance = torch.linalg.norm(sim_pos - expect_pos)
                            if distance.item() < max_distance:
                                results.append(
                                    {
                                        "tnt_0": int(p1),
                                        "tnt_1": int(p2),
                                        "before_end_time": int(p3),
                                        "in_end_time": int(p4),
                                        "distance": float(distance.item()),
                                        "pos": sim_pos.detach().to("cpu").tolist(),
                                    }
                                )
                            progress.update(1)
    finally:
        progress.close()

    return results


if __name__ == "__main__":
    device = base.get_device()
    print("Running on:", device)

    max_tnt = (160, 160)  # (-1,-1) (1,-1)
    max_time_before_end = 20
    max_time_in_end = 80
    max_distance = 20

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

    expect_pos = torch.tensor([1600, 0, 0], dtype=torch.float64, device=device)

    results = pearl_calculation(
        max_tnt,
        max_time_before_end,
        max_time_in_end,
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
    )

    print(f"matches={len(results)}")
    for item in results:
        print(item)
