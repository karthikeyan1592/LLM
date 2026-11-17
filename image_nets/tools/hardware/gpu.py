import time
import torch


class Timer:
    """
    Context manager for wall-clock timing.
    Calls torch.cuda.synchronize() before start and stop when CUDA is available.

    Usage:
        with Timer() as t:
            ...
        print(f"{t.ms:.1f} ms")
    """
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._t0

    @property
    def ms(self):
        return self.elapsed * 1000

    @property
    def s(self):
        return self.elapsed


def reset_memory_stats(device):
    """Reset CUDA peak memory counters.  No-op on CPU."""
    if hasattr(device, "type") and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


if __name__ == "__main__":
    import torch
    with Timer() as t:
        _ = torch.randn(1000, 1000) @ torch.randn(1000, 1000)
    assert t.ms > 0, "FAIL: timer not working"
    print(f"PASS: Timer OK  ({t.ms:.1f} ms for 1000x1000 CPU matmul)")
