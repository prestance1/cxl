from enum import Enum, auto
from dataclasses import dataclass, field
import os


class ComputeBackend(Enum):
    GPU = auto()
    CPU = auto()
    MULTICORE = auto()
    MULTITHREAD = auto()


@dataclass(frozen=True)
class HardwareConfig:
    compute_backend: ComputeBackend
    max_workers: int = field(default_factory=lambda: os.cpu_count())
    no_devices: int = field(default=1)
