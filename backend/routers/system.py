"""System monitoring endpoint."""

import time
from typing import Optional

import psutil
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/system", tags=["system"])

_BOOT_TIME: float = psutil.boot_time()


class SystemStatus(BaseModel):
    """Snapshot of key system resource metrics.

    Attributes:
        cpu_percent: CPU utilisation over the last second (0–100).
        memory_percent: Percentage of RAM in use.
        disk_percent: Percentage of root filesystem in use.
        cpu_temp_celsius: CPU temperature in °C; ``None`` if unavailable on this platform.
        uptime_seconds: Seconds since last boot.
    """

    cpu_percent: float
    memory_percent: float
    disk_percent: float
    cpu_temp_celsius: Optional[float]
    uptime_seconds: float


def _read_cpu_temp() -> Optional[float]:
    """Read the CPU temperature via psutil's ``sensors_temperatures`` if available.

    Returns:
        Temperature in °C or ``None`` if the platform doesn't expose it.
    """
    try:
        temps = psutil.sensors_temperatures()  # type: ignore[attr-defined]
    except AttributeError:
        return None
    if not temps:
        return None
    for key in ("cpu_thermal", "coretemp", "k10temp", "acpitz"):
        entries = temps.get(key)
        if entries:
            return entries[0].current
    # Fall back to the first available sensor
    first = next(iter(temps.values()), None)
    if first:
        return first[0].current
    return None


@router.get("", response_model=SystemStatus)
def get_system() -> SystemStatus:
    """Return a snapshot of current system resource usage.

    Returns:
        ``SystemStatus`` with CPU, memory, disk, temperature, and uptime.
    """
    return SystemStatus(
        cpu_percent=psutil.cpu_percent(interval=1),
        memory_percent=psutil.virtual_memory().percent,
        disk_percent=psutil.disk_usage("/").percent,
        cpu_temp_celsius=_read_cpu_temp(),
        uptime_seconds=time.time() - _BOOT_TIME,
    )
