"""Maintenance hints based on the failure rules documented for the AI4I 2020 dataset.

IMPORTANT: these rules describe how the SYNTHETIC dataset's failure labels were
generated. They are not real-world engineering or safety limits.
"""

import math

HDF_MAX_TEMP_DIFF_K = 8.6  # process minus air temperature
HDF_MAX_SPEED_RPM = 1380
PWF_MIN_WATTS = 3500
PWF_MAX_WATTS = 9000
OSF_LIMITS = {"L": 11000, "M": 12000, "H": 13000}  # tool wear x torque [minNm]
TWF_WEAR_START_MIN = 200  # failures happen at a random wear between 200 and 240 min

RPM_TO_RAD_PER_SECOND = 2 * math.pi / 60


def power_watts(torque_nm: float, speed_rpm: float) -> float:
    """Mechanical power = torque x angular speed (rad/s)."""
    return torque_nm * speed_rpm * RPM_TO_RAD_PER_SECOND


def check_failure_rules(values: dict) -> list[dict]:
    """Return the documented failure rules that the given readings trigger.

    `values` uses the same keys as the model inputs (Type, "Torque [Nm]", ...).
    Each result has: mode, title, why, action.
    """
    air = values["Air temperature [K]"]
    process = values["Process temperature [K]"]
    speed = values["Rotational speed [rpm]"]
    torque = values["Torque [Nm]"]
    wear = values["Tool wear [min]"]
    machine_type = str(values["Type"]).strip().upper()

    triggered = []

    temp_diff = process - air
    if temp_diff < HDF_MAX_TEMP_DIFF_K and speed < HDF_MAX_SPEED_RPM:
        triggered.append({
            "mode": "HDF",
            "title": "Heat dissipation risk",
            "why": (
                f"Process minus air temperature is {temp_diff:.1f} K "
                f"(below {HDF_MAX_TEMP_DIFF_K} K) and speed is {speed:.0f} rpm "
                f"(below {HDF_MAX_SPEED_RPM} rpm)."
            ),
            "action": "Check cooling and ventilation; consider raising the speed or reducing the heat load.",
        })

    power = power_watts(torque, speed)
    if power < PWF_MIN_WATTS or power > PWF_MAX_WATTS:
        side = "below" if power < PWF_MIN_WATTS else "above"
        limit = PWF_MIN_WATTS if power < PWF_MIN_WATTS else PWF_MAX_WATTS
        triggered.append({
            "mode": "PWF",
            "title": "Power out of range",
            "why": f"Power is about {power:,.0f} W, {side} {limit:,} W.",
            "action": "Check the load, drive and power supply for this machine.",
        })

    strain = wear * torque
    limit = OSF_LIMITS.get(machine_type)
    if limit is not None and strain > limit:
        triggered.append({
            "mode": "OSF",
            "title": "Overstrain risk",
            "why": (
                f"Tool wear x torque is {strain:,.0f} minNm, above the {limit:,} "
                f"minNm limit for type {machine_type}."
            ),
            "action": "Reduce the load or replace the tool before continuing.",
        })

    if wear >= TWF_WEAR_START_MIN:
        triggered.append({
            "mode": "TWF",
            "title": "Tool wear risk",
            "why": (
                f"Tool wear is {wear:.0f} min; tools in this dataset fail at a "
                f"random wear between {TWF_WEAR_START_MIN} and 240 min."
            ),
            "action": "Plan a tool replacement soon.",
        })

    return triggered
