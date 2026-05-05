"""
Hardware backend abstraction for AngryBird / WispSim.

Encapsulates device-specific capabilities so the rest of the codebase stays
hardware-agnostic.  The key feature is async fire-estimate dispatch:

  CUDA  — submit_fire_estimate() runs the 1-member fire simulation in a
          dedicated background thread.  The thread has its own CUDA context
          and stream, so the N-member ensemble on the main thread and the
          single-member live estimate can genuinely overlap on the GPU.
          Returns a real concurrent.futures.Future; caller checks
          future.done() each render frame to pick up the result without
          ever blocking the main loop.

  MPS   — PyTorch MPS serialises all Metal command buffers on a shared
          queue; concurrent threads do not yield a speedup and pre-2.5
          PyTorch MPS is not thread-safe across Python threads.
          submit_fire_estimate() runs inline and returns a pre-resolved
          Future, giving MPS the same interface with zero threading overhead.

  CPU   — Same inline path as MPS.

Usage
-----
    from angrybird.hardware import HardwareBackend

    backend = HardwareBackend("cuda")           # or "mps" / "cpu"
    future  = backend.submit_fire_estimate(fn)  # returns immediately on CUDA
    ...                                         # main loop continues
    if future.done():
        result = future.result()                # no blocking — it's ready
    ...
    backend.shutdown()                          # call once at simulation end
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


class HardwareBackend:
    """
    Device capability wrapper for AngryBird compute backends.

    Parameters
    ----------
    device : str
        PyTorch device string: "cuda", "cuda:0", "mps", or "cpu".
        "cuda:N" is normalised to "cuda".

    Attributes
    ----------
    device        : str   Normalised device string.
    supports_async: bool  True only on CUDA with a real GPU present.
    """

    def __init__(self, device: str) -> None:
        # Normalise "cuda:0", "cuda:1", etc. → "cuda"
        base = device.split(":")[0].lower()

        # Validate CUDA availability; fall back gracefully.
        if base == "cuda":
            if not _TORCH_AVAILABLE or not _torch.cuda.is_available():
                logger.warning(
                    "HardwareBackend: CUDA requested but not available "
                    "— scheduling decisions will use CPU path."
                )
                base = "cpu"

        self._device: str = base
        self._supports_async: bool = (base == "cuda")

        # Single worker: one background fire estimate at a time.
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="fire_est")
            if self._supports_async else None
        )
        logger.debug(
            "HardwareBackend ready: device=%s  async=%s",
            self._device, self._supports_async,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> str:
        """Normalised device string: 'cuda', 'mps', or 'cpu'."""
        return self._device

    @property
    def supports_async(self) -> bool:
        """
        True when fire estimates can run concurrently with the main loop.

        CUDA  — the background worker thread has its own CUDA context; the
                main-thread ensemble and the single-member live estimate
                overlap on separate GPU streams.

        MPS   — Metal serialises command buffers on a shared queue; threads
                do not improve throughput and PyTorch MPS is not thread-safe
                prior to version 2.5.

        CPU   — no benefit from threading for serial workloads.
        """
        return self._supports_async

    # ------------------------------------------------------------------
    # Synchronise
    # ------------------------------------------------------------------

    def synchronize(self) -> None:
        """
        Block until all pending GPU work is complete.

        CUDA  — torch.cuda.synchronize()
        MPS   — torch.mps.synchronize() (PyTorch ≥ 2.1)
        CPU   — no-op
        """
        if not _TORCH_AVAILABLE:
            return
        if self._device == "cuda" and _torch.cuda.is_available():
            _torch.cuda.synchronize()
        elif self._device == "mps" and hasattr(_torch, "mps"):
            try:
                _torch.mps.synchronize()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Async fire-estimate dispatch
    # ------------------------------------------------------------------

    def submit_fire_estimate(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> "Future[Any]":
        """
        Submit fn(*args, **kwargs) for execution.

        CUDA  → dispatches to the single-worker ThreadPoolExecutor.
                Returns a Future that resolves when fn() finishes.
                The main simulation loop continues unblocked.

        MPS/CPU → runs fn() inline, wraps the result in an already-
                  resolved Future.  future.done() is immediately True.

        Returns
        -------
        Future
            Resolves to fn()'s return value, or holds an exception if
            fn() raised one (surfaced via future.result()).
        """
        if self._supports_async and self._executor is not None:
            return self._executor.submit(fn, *args, **kwargs)

        # Blocking path — run inline, return a pre-resolved Future so the
        # caller can use a single code path regardless of device.
        f: Future = Future()
        f.set_running_or_notify_cancel()
        try:
            f.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001
            f.set_exception(exc)
        return f

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, wait: bool = True) -> None:
        """
        Gracefully shut down the background executor.

        Should be called once at the end of a simulation run (e.g. in the
        finally block of SimulationRunner.run()).  Safe to call multiple
        times.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None

    def __repr__(self) -> str:
        return (
            f"HardwareBackend(device={self._device!r}, "
            f"supports_async={self._supports_async})"
        )
