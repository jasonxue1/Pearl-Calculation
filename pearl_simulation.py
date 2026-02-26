import numpy as np
from numpy.typing import ArrayLike, NDArray

import mth

g = np.float64(0.03)
d = np.float32(0.99)

END_SPAWN_POS = np.array([100.5, 50, 0.5], dtype=np.float64)


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

    if tick == 0:
        return pos

    log and log_state(0, pos, vel, yaw)

    for current_tick in range(1, tick + 1):
        teleport = 1 if current_tick == to_end_time else 0
        pos, vel, yaw = simulate_tick(pos, vel, yaw, teleport)

        log and log_state(current_tick, pos, vel, yaw)

    return pos


def log_state(
    tick: int, pos: NDArray[np.float64], vel: NDArray[np.float64], yaw: ArrayLike
) -> None:
    pos_list = pos.tolist()
    vel_list = vel.tolist()
    yaw_val = float(yaw)
    print(f"tick={tick} pos={pos_list} vel={vel_list} yaw={yaw_val}")


if __name__ == "__main__":
    tnt_count = np.array([97, 56], dtype=np.int32)

    pearl_position = np.array([0, 252.71360805009243, 0], dtype=np.float64)

    pearl_motion = np.array([0, 0.3827286093776437, 0], dtype=np.float64)

    tnt_motion_per_tnt = np.array(
        [0.6406475114548377, 0.0000041762421424, 0.6406475114548377], dtype=np.float64
    )

    tick = 2

    to_end_time = 1

    pos = simulate_pearl(
        tick,
        pearl_position,
        pearl_motion,
        tnt_motion_per_tnt,
        tnt_count,
        to_end_time,
        True,
    )

    print(f"final_pos={pos.tolist()}")
