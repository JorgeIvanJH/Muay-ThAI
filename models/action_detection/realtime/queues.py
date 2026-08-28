"""Bounded queue operations with explicit real-time overload behaviour."""

from __future__ import annotations

import queue
import threading
from typing import TypeVar


ItemT = TypeVar("ItemT")


def put_latest(target: queue.Queue[ItemT], item: ItemT) -> int:
    """Insert an item, discarding the oldest queued item when necessary.

    Usage: Inference only.

    Returns the number of discarded items. A bounded latest-value queue keeps
    webcam latency stable when a downstream stage briefly falls behind.
    """

    dropped = 0
    while True:
        try:
            target.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                target.get_nowait()
                dropped += 1
            except queue.Empty:
                # Another consumer made room between ``put`` and ``get``.
                continue


def put_ordered(
    target: queue.Queue[ItemT],
    item: ItemT,
    stop_event: threading.Event,
) -> bool:
    """Insert an ordered item while remaining responsive to shutdown.

    Usage: Inference only.

    Returns ``False`` if shutdown was requested before the item could be
    inserted. File inference uses this operation because it must not drop
    source frames.
    """

    while not stop_event.is_set():
        try:
            target.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False
