from __future__ import annotations

from typing import Dict, List, Tuple


# alias -> blob key
OPTIONAL_CHANNEL_ALIASES: Dict[str, str] = {
    "surface_tp_6h": "surface_tp_6h",
    "vertical_velocity_input": "pangu_w",
    "surface_geopotential": "surface_z_at_surface",
    "land_sea_mask": "land_sea_mask",
    "precipitation_input": "surface_tp_6h",
}


def model_optional_channels(model_name: str) -> List[str]:
    m = model_name.lower()
    if m == "fuxi":
        return ["surface_tp_6h"]
    if m in ("graphcast_official_operational", "graphcast_official_operational_stepwise"):
        return [
            "vertical_velocity_input",
            "precipitation_input",
            "surface_geopotential",
            "land_sea_mask",
        ]
    return []


def decide_optional_channels(
    model_name: str,
    blob_keys: List[str],
    is_available_cb,
) -> Tuple[List[str], List[str]]:
    enabled: List[str] = []
    missing: List[str] = []
    bkeys = set(blob_keys)
    for alias in model_optional_channels(model_name):
        key = OPTIONAL_CHANNEL_ALIASES[alias]
        if key in bkeys or bool(is_available_cb(alias)):
            enabled.append(alias)
        else:
            missing.append(alias)
    return enabled, missing

