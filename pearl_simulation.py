import torch

import base
import mth


def simulate_tick(
        pos,
        vel,
        yaw,
        teleport,
        GRAVITY,
        DRAG,
        RADIANS_TO_DEGREES,
        DEGREES_TO_RADIANS,
        SIN,
        SCALE,
        COS_OFFSET,
):
    """
    teleport
    0 -> none
    -1 -> to overworld
    1 -> to end
    """
    vel[1] -= GRAVITY
    vel *= DRAG
    target_yaw = (torch.atan2(vel[0], vel[2]) * RADIANS_TO_DEGREES).to(torch.float32)
    yaw = yaw + 0.2 * mth.wrap_degrees(target_yaw - yaw)

    match teleport:
        case 0:
            pos += vel
        case 1:
            old_yaw = yaw
            yaw = torch.tensor(90, dtype=torch.float32, device=SIN.device)
            vel = mth.rotate_yaw_vector(
                vel, old_yaw, yaw, DEGREES_TO_RADIANS, SIN, SCALE, COS_OFFSET
            )
            pos = torch.tensor([100.5, 50, 0.5], dtype=torch.float64, device=SIN.device)
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
    tnt_num = torch.tensor([a+b, abs(a) + abs(b),
                            a - b], dtype=torch.int32, device=tnt_motion.device)

    vel = tnt_num * tnt_motion + pearl_motion
    yaw = torch.tensor(0, dtype=torch.float32, device=tnt_motion.device)
    return pearl_position, vel, yaw


def simulate_pearl(
        tick,
        to_end_time=0,
        log=False,
        pearl_position=None,
        pearl_motion=None,
        tnt_motion=None,
        tnt_count=None,
        GRAVITY=None,
        DRAG=None,
        RADIANS_TO_DEGREES=None,
        DEGREES_TO_RADIANS=None,
        SIN=None,
        SCALE=None,
        COS_OFFSET=None,
):
    if tick < 0:
        raise ValueError("tick must be a nonnegative integer")
    if to_end_time >= tick:
        raise ValueError("to_end_time must be less than tick")
    if to_end_time < 0:
        raise ValueError("to_end_time must be a nonnegative integer")

    pos, vel, yaw = setup_pearl(pearl_position, pearl_motion, tnt_motion, tnt_count)

    log_fn = None
    if log:
        log_fn = log_state
    elif callable(log):
        log_fn = log

    if log_fn:
        log_fn(0, pos, vel, yaw)

    if tick == 0:
        return pos

    for current_tick in range(1, tick + 1):
        teleport = 1 if current_tick == to_end_time else 0
        pos, vel, yaw = simulate_tick(
            pos,
            vel,
            yaw,
            teleport,
            GRAVITY,
            DRAG,
            RADIANS_TO_DEGREES,
            DEGREES_TO_RADIANS,
            SIN,
            SCALE,
            COS_OFFSET,
        )
        if log_fn:
            log_fn(current_tick, pos, vel, yaw)

    return pos


def log_state(tick, pos, vel, yaw):
    pos_list = pos.detach().to("cpu").tolist()
    vel_list = vel.detach().to("cpu").tolist()
    yaw_val = float(yaw.detach().to("cpu"))
    print(f"tick={tick} pos={pos_list} vel={vel_list} yaw={yaw_val}")


if __name__ == "__main__":
    device = base.get_device()
    print("Running on:", device)

    tnt_count = torch.tensor([100, 134], dtype=torch.int32, device=device)

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

    pos = simulate_pearl(
        30,
        8,
        True,
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

    print(f"final_pos={pos.detach().to('cpu').tolist()}")
