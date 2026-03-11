"""
SysInfoTool – reads CPU, RAM, and storage metrics from the local Linux host.

Security guarantees
───────────────────
* Uses `psutil` exclusively – ZERO subprocess / shell calls.
* No user-supplied data is ever passed to a shell, command-line argument,
  file path, or format string that reaches the OS.  All values returned by
  psutil are numeric or simple strings produced by the OS itself, not by
  the user, so prompt-injection via crafted input is structurally impossible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import psutil

logger = logging.getLogger(__name__)


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class CpuInfo:
    physical_cores: int
    logical_cores:  int
    usage_percent:  float          # 1-second average, all cores
    freq_mhz:       float | None   # current frequency, None if unavailable


@dataclass(frozen=True, slots=True)
class RamInfo:
    total_gb:     float
    used_gb:      float
    available_gb: float
    percent:      float


@dataclass(frozen=True, slots=True)
class DiskPartitionInfo:
    mountpoint: str
    fstype:     str
    total_gb:   float
    used_gb:    float
    free_gb:    float
    percent:    float


@dataclass(frozen=True, slots=True)
class SysInfoResult:
    cpu:   CpuInfo
    ram:   RamInfo
    disks: list[DiskPartitionInfo] = field(default_factory=list)

    def as_text(self) -> str:
        """Return a human-readable summary suitable for injection into an LLM prompt."""
        lines: list[str] = []

        # CPU
        freq_str = (
            f"{self.cpu.freq_mhz:.0f} MHz"
            if self.cpu.freq_mhz is not None
            else "N/A"
        )
        lines += [
            "=== CPU ===",
            f"  Physical cores : {self.cpu.physical_cores}",
            f"  Logical cores  : {self.cpu.logical_cores}",
            f"  Current freq   : {freq_str}",
            f"  Usage (1s avg) : {self.cpu.usage_percent:.1f}%",
        ]

        # RAM
        lines += [
            "",
            "=== RAM ===",
            f"  Total     : {self.ram.total_gb:.2f} GB",
            f"  Used      : {self.ram.used_gb:.2f} GB",
            f"  Available : {self.ram.available_gb:.2f} GB",
            f"  Usage     : {self.ram.percent:.1f}%",
        ]

        # Disks
        lines += ["", "=== Storage ==="]
        if self.disks:
            for d in self.disks:
                lines += [
                    f"  [{d.mountpoint}]  ({d.fstype})",
                    f"    Total : {d.total_gb:.2f} GB",
                    f"    Used  : {d.used_gb:.2f} GB",
                    f"    Free  : {d.free_gb:.2f} GB",
                    f"    Usage : {d.percent:.1f}%",
                ]
        else:
            lines.append("  (no partitions found)")

        return "\n".join(lines)


# ── Tool ─────────────────────────────────────────────────────────────────────

class SysInfoTool:
    """
    Collects hardware resource metrics via psutil.

    No user input is accepted or processed; the tool purely reads OS-level
    metrics, so there is no attack surface for prompt injection.
    """

    _BYTES_PER_GB: float = 1024 ** 3

    def collect(self) -> SysInfoResult:
        """
        Gather CPU, RAM, and disk information synchronously.

        Returns
        -------
        SysInfoResult
            Structured snapshot of current resource usage.
        """
        logger.debug("SysInfoTool: collecting system metrics via psutil")

        cpu  = self._collect_cpu()
        ram  = self._collect_ram()
        disks = self._collect_disks()

        logger.debug(
            "SysInfoTool: cpu=%.1f%% ram=%.1f%% partitions=%d",
            cpu.usage_percent, ram.percent, len(disks),
        )
        return SysInfoResult(cpu=cpu, ram=ram, disks=disks)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _collect_cpu(self) -> CpuInfo:
        physical = psutil.cpu_count(logical=False) or 0
        logical  = psutil.cpu_count(logical=True)  or 0
        usage    = psutil.cpu_percent(interval=1)   # blocks ~1 s for accuracy

        freq_info = psutil.cpu_freq()
        freq_mhz  = freq_info.current if freq_info else None

        return CpuInfo(
            physical_cores=physical,
            logical_cores=logical,
            usage_percent=float(usage),
            freq_mhz=float(freq_mhz) if freq_mhz is not None else None,
        )

    def _collect_ram(self) -> RamInfo:
        mem = psutil.virtual_memory()
        gb  = self._BYTES_PER_GB
        return RamInfo(
            total_gb=mem.total     / gb,
            used_gb=mem.used       / gb,
            available_gb=mem.available / gb,
            percent=float(mem.percent),
        )

    def _collect_disks(self) -> list[DiskPartitionInfo]:
        """Return usage stats for all mounted physical partitions."""
        result: list[DiskPartitionInfo] = []
        gb = self._BYTES_PER_GB

        for part in psutil.disk_partitions(all=False):
            # Skip pseudo / virtual filesystems
            if not part.device or part.fstype in {"", "tmpfs", "devtmpfs", "squashfs"}:
                continue
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except (PermissionError, OSError) as exc:
                logger.warning(
                    "SysInfoTool: cannot read usage for %s – %s", part.mountpoint, exc
                )
                continue

            result.append(
                DiskPartitionInfo(
                    mountpoint=part.mountpoint,
                    fstype=part.fstype,
                    total_gb=usage.total / gb,
                    used_gb=usage.used   / gb,
                    free_gb=usage.free   / gb,
                    percent=float(usage.percent),
                )
            )

        return result
