"""Network monitoring + on-demand internet speed-test endpoints.

Two distinct capabilities live here:

* **Passive usage graph** — a background daemon thread samples the host's NIC
  byte counters (``psutil.net_io_counters``) every few seconds into an in-memory
  ring buffer.  ``GET /api/network/history`` returns the per-sample download /
  upload throughput so the frontend can plot it over a selectable time range.
  Reading the counters transfers nothing over the network, so this sampler is
  free to run continuously.

* **Active speed test** — ``POST /api/network/speedtest`` downloads a small
  payload from, and uploads a small payload to, Cloudflare's public speed-test
  endpoint and reports the measured Mbps each way.  This *does* consume
  bandwidth, so the payloads are kept deliberately small (the Pi has a limited
  connection) and the test runs only on demand.

Environment variables:
    SPEEDTEST_DOWNLOAD_BYTES: Bytes to download per test (default ~1 MB).
    SPEEDTEST_UPLOAD_BYTES: Bytes to upload per test (default ~256 KB).
    SPEEDTEST_DOWN_URL / SPEEDTEST_UP_URL: Override the speed-test endpoints.
"""

import os
import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import httpx
import psutil
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/network", tags=["network"])


# ---------------------------------------------------------------------------
# Passive usage sampler
# ---------------------------------------------------------------------------

_SAMPLE_INTERVAL_SEC = 3.0
# Retain ~1 hour of history at the sample interval above (plus a little slack).
_HISTORY_RETENTION_SEC = 3600
_MAX_SAMPLES = int(_HISTORY_RETENTION_SEC / _SAMPLE_INTERVAL_SEC) + 10

# Selectable history windows exposed to the frontend, mapped to seconds.
_RANGE_SECONDS = {"5m": 300, "30m": 1800, "1h": 3600}


class NetworkSample(BaseModel):
    """One network-throughput sample.

    Attributes:
        t: Unix timestamp (seconds) the sample was taken.
        rx_kbps: Download rate since the previous sample, in kilobits/sec.
        tx_kbps: Upload rate since the previous sample, in kilobits/sec.
    """

    t: float
    rx_kbps: float
    tx_kbps: float


class NetworkHistory(BaseModel):
    """A window of throughput samples.

    Attributes:
        interval_sec: Nominal seconds between samples (the sampler cadence).
        samples: Samples within the requested window, oldest first.
    """

    interval_sec: float
    samples: List[NetworkSample]


class _Sampler:
    """Background thread that records NIC throughput into a ring buffer.

    Each tick reads the cumulative byte counters and derives a per-second rate
    from the delta against the previous reading, so the very first reading only
    seeds the baseline (it produces no sample).
    """

    def __init__(self, interval: float, maxlen: int) -> None:
        """Initialise the sampler.

        Args:
            interval: Seconds to sleep between samples.
            maxlen: Maximum number of samples to retain (ring-buffer size).
        """
        self._interval = interval
        self._samples: Deque[NetworkSample] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._prev: Optional[Tuple[float, int, int]] = None
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def start(self) -> None:
        """Start the background sampler thread once (idempotent)."""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._run, name="net-sampler", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        """Sample forever on the configured interval."""
        while True:
            try:
                self._sample_once()
            except Exception:  # pylint: disable=broad-exception-caught
                # A transient counter read failure must never kill the sampler.
                pass
            time.sleep(self._interval)

    def _sample_once(self) -> None:
        """Take one reading and append a derived-rate sample (after the first)."""
        counters = psutil.net_io_counters()
        now = time.time()
        recv, sent = int(counters.bytes_recv), int(counters.bytes_sent)
        if self._prev is not None:
            prev_t, prev_recv, prev_sent = self._prev
            elapsed = now - prev_t
            if elapsed > 0:
                rx_kbps = max(0, recv - prev_recv) * 8 / elapsed / 1000
                tx_kbps = max(0, sent - prev_sent) * 8 / elapsed / 1000
                with self._lock:
                    self._samples.append(
                        NetworkSample(t=now, rx_kbps=rx_kbps, tx_kbps=tx_kbps)
                    )
        self._prev = (now, recv, sent)

    def history(self, since_sec: float) -> List[NetworkSample]:
        """Return samples taken within the last ``since_sec`` seconds.

        Args:
            since_sec: Window length in seconds.

        Returns:
            Samples newer than the cutoff, oldest first.
        """
        cutoff = time.time() - since_sec
        with self._lock:
            return [s for s in self._samples if s.t >= cutoff]


# A single module-level sampler started at import, mirroring the module-global
# pattern used elsewhere in the backend (e.g. ``system._BOOT_TIME``).
_sampler = _Sampler(_SAMPLE_INTERVAL_SEC, _MAX_SAMPLES)
_sampler.start()


@router.get("/history", response_model=NetworkHistory)
def get_history(window: str = Query("5m", alias="range")) -> NetworkHistory:
    """Return NIC throughput samples for the requested time window.

    Args:
        window: One of ``5m``, ``30m``, ``1h`` (query param ``range``).

    Returns:
        ``NetworkHistory`` with the sampler cadence and the matching samples.

    Raises:
        HTTPException: 400 when ``range`` is not a recognised window.
    """
    since = _RANGE_SECONDS.get(window)
    if since is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid range '{window}'. Use one of: {', '.join(_RANGE_SECONDS)}",
        )
    return NetworkHistory(
        interval_sec=_SAMPLE_INTERVAL_SEC, samples=_sampler.history(since)
    )


# ---------------------------------------------------------------------------
# Active speed test
# ---------------------------------------------------------------------------

_DOWNLOAD_BYTES = int(os.environ.get("SPEEDTEST_DOWNLOAD_BYTES", 1_048_576))  # ~1 MB
_UPLOAD_BYTES = int(os.environ.get("SPEEDTEST_UPLOAD_BYTES", 262_144))  # ~256 KB
_SPEEDTEST_DOWN_URL = os.environ.get(
    "SPEEDTEST_DOWN_URL", "https://speed.cloudflare.com/__down"
)
_SPEEDTEST_UP_URL = os.environ.get(
    "SPEEDTEST_UP_URL", "https://speed.cloudflare.com/__up"
)
_SPEEDTEST_TIMEOUT_SEC = 30.0


class SpeedTestResult(BaseModel):
    """Result of one on-demand internet speed test.

    Attributes:
        download_mbps: Measured download throughput in megabits/sec.
        upload_mbps: Measured upload throughput in megabits/sec.
        download_bytes: Bytes actually downloaded during the test.
        upload_bytes: Bytes actually uploaded during the test.
        ran_at: Unix timestamp (seconds) the test completed.
    """

    download_mbps: float
    upload_mbps: float
    download_bytes: int
    upload_bytes: int
    ran_at: float


def _mbps(num_bytes: int, elapsed_sec: float) -> float:
    """Convert a byte count + duration to megabits/sec.

    Args:
        num_bytes: Number of bytes transferred.
        elapsed_sec: Seconds the transfer took.

    Returns:
        Throughput in megabits/sec, or 0.0 when ``elapsed_sec`` is non-positive.
    """
    if elapsed_sec <= 0:
        return 0.0
    return num_bytes * 8 / elapsed_sec / 1_000_000


def run_speed_test() -> SpeedTestResult:
    """Measure download + upload throughput against the speed-test endpoint.

    Downloads ``_DOWNLOAD_BYTES`` then uploads ``_UPLOAD_BYTES`` (both small, to
    spare a limited connection) and times each leg.

    Returns:
        A ``SpeedTestResult`` with the measured Mbps each way.

    Raises:
        httpx.HTTPError: If either leg fails (relayed as a 503 by the route).
    """
    with httpx.Client(timeout=_SPEEDTEST_TIMEOUT_SEC) as client:
        start = time.perf_counter()
        resp = client.get(_SPEEDTEST_DOWN_URL, params={"bytes": _DOWNLOAD_BYTES})
        resp.raise_for_status()
        downloaded = len(resp.content)
        download_elapsed = time.perf_counter() - start

        payload = b"\0" * _UPLOAD_BYTES
        start = time.perf_counter()
        up_resp = client.post(_SPEEDTEST_UP_URL, content=payload)
        up_resp.raise_for_status()
        upload_elapsed = time.perf_counter() - start

    return SpeedTestResult(
        download_mbps=_mbps(downloaded, download_elapsed),
        upload_mbps=_mbps(_UPLOAD_BYTES, upload_elapsed),
        download_bytes=downloaded,
        upload_bytes=_UPLOAD_BYTES,
        ran_at=time.time(),
    )


@router.post("/speedtest", response_model=SpeedTestResult)
def speed_test() -> SpeedTestResult:
    """Run an on-demand internet speed test and return the result.

    Returns:
        ``SpeedTestResult`` with measured download/upload Mbps.

    Raises:
        HTTPException: 503 when the speed-test endpoint is unreachable.
    """
    try:
        return run_speed_test()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"Speed test failed: {exc}"
        ) from exc
