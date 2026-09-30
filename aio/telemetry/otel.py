"""
AIO v3 – Feature 11: Télémétrie OpenTelemetry / Prometheus
Export /metrics compatible Prometheus sans dépendance externe.
"""
from __future__ import annotations
import time
from typing import Dict, Any, List


class MetricsRegistry:
    """Registre de métriques compatibles Prometheus (format text/plain)."""

    def __init__(self):
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._labels: Dict[str, Dict[str, str]] = {}
        self._start = time.time()

    def counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        key = self._key(name, labels)
        self._counters[key] = self._counters.get(key, 0.0) + value
        if labels:
            self._labels[key] = labels

    def gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        key = self._key(name, labels)
        self._gauges[key] = value
        if labels:
            self._labels[key] = labels

    def histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        key = self._key(name, labels)
        self._histograms.setdefault(key, []).append(value)
        if labels:
            self._labels[key] = labels

    def _key(self, name: str, labels: Dict[str, str] = None) -> str:
        if not labels:
            return name
        lbl = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{lbl}}}"

    def _format_labels(self, key: str) -> str:
        if "{" in key:
            name, rest = key.split("{", 1)
            return key
        return key

    def export_prometheus(self) -> str:
        """Génère le texte au format Prometheus text/plain."""
        lines = [
            f"# AIO v3 metrics – uptime {time.time()-self._start:.0f}s",
            "",
        ]
        # Counters
        for key, val in self._counters.items():
            base = key.split("{")[0]
            lines.append(f"# TYPE {base} counter")
            lines.append(f"{key} {val}")
        # Gauges
        for key, val in self._gauges.items():
            base = key.split("{")[0]
            lines.append(f"# TYPE {base} gauge")
            lines.append(f"{key} {val}")
        # Histograms (sum + count + quantiles)
        for key, values in self._histograms.items():
            base = key.split("{")[0]
            lbl_part = ("{" + key.split("{", 1)[1]) if "{" in key else ""
            sorted_v = sorted(values)
            n = len(sorted_v)
            total = sum(sorted_v)

            def quantile(q):
                idx = max(0, min(int(q * n) - 1, n - 1))
                return sorted_v[idx] if sorted_v else 0.0

            lines.append(f"# TYPE {base} histogram")
            for q in [0.5, 0.9, 0.99]:
                q_lbl = lbl_part.rstrip("}") + f',quantile="{q}"}}' if lbl_part else f'{{quantile="{q}"}}'
                lines.append(f'{base}{q_lbl} {quantile(q):.3f}')
            lines.append(f"{base}_sum{lbl_part} {total:.3f}")
            lines.append(f"{base}_count{lbl_part} {n}")
        return "\n".join(lines) + "\n"

    def snapshot(self) -> Dict[str, Any]:
        """Snapshot JSON pour le dashboard."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                k: {
                    "count": len(v),
                    "sum": sum(v),
                    "p50": sorted(v)[len(v)//2] if v else 0,
                    "p99": sorted(v)[int(len(v)*0.99)] if v else 0,
                }
                for k, v in self._histograms.items()
            },
            "uptime_s": round(time.time() - self._start, 1),
        }


# Registre global partagé
METRICS = MetricsRegistry()
