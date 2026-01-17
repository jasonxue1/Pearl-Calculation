import torch
from typing import Tuple

Tensor = torch.Tensor


def build_lut(device: torch.device) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    RADIANS_TO_DEGREES = torch.tensor(
        57.2957763671875, dtype=torch.float64, device=device
    )
    DEGREES_TO_RADIANS = torch.tensor(
        0.017453292519943295, dtype=torch.float64, device=device
    )

    SCALE = torch.tensor(10430.378, dtype=torch.float32, device=device)
    COS_OFFSET = torch.tensor(16384, dtype=torch.float32, device=device)

    i = torch.arange(65536, dtype=torch.float64, device=device)
    SIN = torch.sin(
        i * torch.tensor(torch.pi, dtype=torch.float64, device=device) * 2.0 / 65536.0
    )
    SIN = SIN.to(torch.float32)

    return SIN, SCALE, COS_OFFSET, RADIANS_TO_DEGREES, DEGREES_TO_RADIANS


def sin(v: Tensor, SIN: Tensor, SCALE: Tensor) -> Tensor:
    v = torch.as_tensor(v, dtype=torch.float32, device=SIN.device)
    idx = (v * SCALE).to(torch.int32) & 65535
    return SIN[idx]


def cos(v: Tensor, SIN: Tensor, SCALE: Tensor, COS_OFFSET: Tensor) -> Tensor:
    v = torch.as_tensor(v, dtype=torch.float32, device=SIN.device)
    idx = (v * SCALE + COS_OFFSET).to(torch.int32) & 65535
    return SIN[idx]


def wrap_degrees(deg: Tensor) -> Tensor:
    d = deg.to(dtype=torch.float32)
    return torch.remainder(d + 180.0, 360.0) - 180.0


def rotate_yaw_vector(
    vec: Tensor,
    old_yaw: Tensor,
    new_yaw: Tensor,
    DEGREES_TO_RADIANS: Tensor,
    SIN: Tensor,
    SCALE: Tensor,
    COS_OFFSET: Tensor,
) -> Tensor:
    yaw_delta = torch.as_tensor(
        old_yaw - new_yaw, dtype=torch.float32, device=SIN.device
    )
    rad = yaw_delta * DEGREES_TO_RADIANS

    c = cos(rad, SIN, SCALE, COS_OFFSET)
    s = sin(rad, SIN, SCALE)

    x, y, z = vec.unbind(dim=-1)

    out = torch.stack([x * c + z * s, y, z * c - x * s], dim=-1)
    return out
