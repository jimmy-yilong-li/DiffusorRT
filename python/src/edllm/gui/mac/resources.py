"""What a Mac window shows of memory. It never creates a backend or owns device resources.

author:Opus
"""
from __future__ import annotations

import resource
import sys
import time
from typing import Any


def snapshot(engine: Any | None) -> dict[str, Any]:
    """MLX allocator readings from the open engine, and this process's lifetime peak."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    reading: dict[str, Any] = {
        "platform": sys.platform, "sampled_at": time.time(),
        "mlx_active_bytes": None, "mlx_cache_bytes": None,
        # macOS reports bytes; the other Unixes this may be tested on report kilobytes.
        "process_peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
    }
    if engine is not None:
        reading.update(engine.resource_snapshot())
    return reading
