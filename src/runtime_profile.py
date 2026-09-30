"""Low-overhead wall/CPU/memory timings for long prediction runs.

The profiler writes a tiny JSON file after every completed stage. A multi-hour
run therefore leaves useful evidence even if a later model fails, without
serialising dataframes or retaining another copy of the training matrix.
"""
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import sys
import time


def _rss_mb():
    # Linux reports ru_maxrss in KiB. This repository's supported long-run
    # environment is Linux; keep the conversion explicit in the JSON schema.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _sync_neural_gpu():
    """Make GPU-only neural stages report completed work, not queued kernels."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


class RuntimeProfile:
    """Collect ordered stage timings and optionally persist them as JSON."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.enabled = self.path is not None
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started_wall = time.perf_counter()
        self.stages = []

    def stage(self, name, **details):
        if not self.enabled:
            return nullcontext()
        return self._stage(name, details)

    @contextmanager
    def _stage(self, name, details):
        _sync_neural_gpu()
        wall0, cpu0, rss0 = time.perf_counter(), time.process_time(), _rss_mb()
        print(f"[timing] start {name}", flush=True)
        try:
            yield
        finally:
            _sync_neural_gpu()
            wall = time.perf_counter() - wall0
            cpu = time.process_time() - cpu0
            record = {
                "name": name,
                "wall_seconds": round(wall, 6),
                "cpu_seconds": round(cpu, 6),
                "cpu_over_wall": round(cpu / wall, 3) if wall else None,
                "max_rss_mb_before": round(rss0, 3),
                "max_rss_mb_after": round(_rss_mb(), 3),
                **details,
            }
            self.stages.append(record)
            print(f"[timing] done  {name}: wall={wall:.2f}s cpu={cpu:.2f}s "
                  f"max_rss={record['max_rss_mb_after']:.0f}MB", flush=True)
            self.flush()

    def flush(self, finished=False):
        if not self.enabled:
            return
        payload = {
            "schema_version": 1,
            "pid": os.getpid(),
            "started_at_utc": self.started_at,
            "finished": bool(finished),
            "elapsed_wall_seconds": round(time.perf_counter() - self.started_wall, 6),
            "rss_unit": "MiB (Linux ru_maxrss)",
            "stages": self.stages,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(self.path)

    def finish(self):
        if not self.enabled:
            return
        self.flush(finished=True)
        accounted = sum(s["wall_seconds"] for s in self.stages)
        elapsed = time.perf_counter() - self.started_wall
        print(f"[timing] total wall={elapsed:.2f}s; timed stages={accounted:.2f}s; "
              f"report={self.path}", flush=True)
