"""Confidence Cascade — Three-zone confidence propagation (GREEN/YELLOW/RED).

Track confidence degradation through sequential processes, combine
multiple signals with weighted importance, and automatically escalate.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class ConfidenceZone(Enum):
    GREEN = "green"      # 0.85-1.0: Auto-proceed
    YELLOW = "yellow"    # 0.60-0.85: Proceed with caution
    RED = "red"          # 0.00-0.60: Stop and investigate


@dataclass
class Confidence:
    """A confidence value with source tracking."""
    value: float  # 0.0 to 1.0
    source: str
    zone: ConfidenceZone = ConfidenceZone.GREEN
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.zone = self._classify()
    
    def _classify(self) -> ConfidenceZone:
        if self.value >= 0.85:
            return ConfidenceZone.GREEN
        elif self.value >= 0.60:
            return ConfidenceZone.YELLOW
        return ConfidenceZone.RED
    
    def degrade(self, factor: float) -> "Confidence":
        """Degrade confidence by a factor (0-1, where 0 = total loss)."""
        new_val = max(0, self.value * (1 - factor))
        return Confidence(new_val, self.source, weight=self.weight, metadata=self.metadata.copy())
    
    def boost(self, factor: float) -> "Confidence":
        """Boost confidence by a factor."""
        new_val = min(1.0, self.value + factor * (1.0 - self.value))
        return Confidence(new_val, self.source, weight=self.weight, metadata=self.metadata.copy())


class CascadeConfig:
    """Configuration for cascade thresholds and behavior."""
    
    def __init__(self, green_threshold: float = 0.85, yellow_threshold: float = 0.60,
                 degradation_rate: float = 0.05, min_confidence: float = 0.1):
        self.green_threshold = green_threshold
        self.yellow_threshold = yellow_threshold
        self.degradation_rate = degradation_rate
        self.min_confidence = min_confidence


def create_confidence(value: float, source: str, weight: float = 1.0) -> Confidence:
    return Confidence(value=value, source=source, weight=weight)


def sequential_cascade(confidences: List[Confidence], config: CascadeConfig = None) -> Tuple[Confidence, List[Confidence]]:
    """Run confidence through sequential stages. Each stage degrades.
    
    Like a fishing net — if any mesh tears, the whole thing's compromised.
    """
    config = config or CascadeConfig()
    stages = []
    running = confidences[0] if confidences else Confidence(1.0, "empty")
    
    for i, c in enumerate(confidences):
        # Each stage applies degradation proportional to weakness
        weakness = max(0, config.yellow_threshold - c.value) / config.yellow_threshold
        degradation = weakness * config.degradation_rate * (i + 1)
        running = running.degrade(degradation)
        
        # If any stage is RED, major cascade failure
        if c.zone == ConfidenceZone.RED:
            running = Confidence(
                max(running.value - 0.1 * (i + 1), config.min_confidence),
                running.source, weight=running.weight)
        
        stages.append(running)
    
    # Apply all weights
    for c in confidences:
        running = Confidence(
            running.value * (0.5 + 0.5 * c.value),
            f"cascade({','.join(c.source for c in confidences)})",
            weight=1.0)
    
    return (running, stages)


def parallel_cascade(signals: List[Dict[str, Any]], config: CascadeConfig = None) -> Tuple[Confidence, Dict[str, Confidence]]:
    """Combine multiple signals with weights in parallel.
    
    Args:
        signals: [{"confidence": Confidence, "weight": float}, ...]
    Returns:
        (combined_confidence, {"zone": zone_counts})
    """
    config = config or CascadeConfig()
    
    if not signals:
        return (Confidence(0.5, "none"), {})
    
    total_weight = sum(s["weight"] for s in signals)
    if total_weight == 0:
        total_weight = 1.0
    
    # Weighted average with penalty for outliers
    values = [(s["confidence"].value, s["weight"]) for s in signals]
    weighted_sum = sum(v * w for v, w in values)
    weighted_avg = weighted_sum / total_weight
    
    # Penalty: any RED signal drags down the whole
    red_count = sum(1 for s in signals if s["confidence"].zone == ConfidenceZone.RED)
    yellow_count = sum(1 for s in signals if s["confidence"].zone == ConfidenceZone.YELLOW)
    green_count = sum(1 for s in signals if s["confidence"].zone == ConfidenceZone.GREEN)
    
    if red_count > 0:
        penalty = 0.1 * red_count
        weighted_avg = max(weighted_avg - penalty, config.min_confidence)
    elif yellow_count > len(signals) / 2:
        penalty = 0.05 * yellow_count
        weighted_avg = max(weighted_avg - penalty, config.min_confidence)
    
    combined = Confidence(
        weighted_avg,
        f"parallel({','.join(s['confidence'].source for s in signals)})",
        weight=1.0
    )
    
    zone_counts = {"green": green_count, "yellow": yellow_count, "red": red_count}
    return (combined, zone_counts)


def conditional_cascade(confidence: Confidence, routes: Dict[str, Callable[[float], Confidence]],
                        config: CascadeConfig = None) -> Tuple[Confidence, str]:
    """Route through different validation paths based on zone.
    
    Args:
        confidence: Input confidence
        routes: {zone_name: function that takes confidence and returns Confidence}
    Returns:
        (output_confidence, route_taken)
    """
    config = config or CascadeConfig()
    route_name = confidence.zone.value
    route_fn = routes.get(route_name)
    
    if route_fn:
        result = route_fn(confidence.value)
    else:
        result = confidence
    
    return (result, route_name)


class ConfidenceTracker:
    """Track confidence over time with history and trends."""
    
    def __init__(self, source: str, max_history: int = 100):
        self.source = source
        self._history: List[Tuple[int, float, ConfidenceZone]] = []
        self._max_history = max_history
    
    def record(self, value: float, tick: int = 0):
        c = Confidence(value, self.source)
        self._history.append((tick, value, c.zone))
        if len(self._history) > self._max_history:
            self._history.pop(0)
    
    def trend(self, window: int = 10) -> float:
        """Calculate trend: positive = improving, negative = degrading."""
        if len(self._history) < 2:
            return 0.0
        recent = [v for _, v, _ in self._history[-window:]]
        if len(recent) < 2:
            return 0.0
        return (recent[-1] - recent[0]) / len(recent)
    
    def current(self) -> Optional[Confidence]:
        if self._history:
            _, val, _ = self._history[-1]
            return Confidence(val, self.source)
        return None
    
    def zone_history(self) -> List[ConfidenceZone]:
        return [z for _, _, z in self._history]
    
    def stats(self) -> dict:
        if not self._history:
            return {"source": self.source, "samples": 0}
        values = [v for _, v, _ in self._history]
        return {
            "source": self.source,
            "samples": len(self._history),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "trend": self.trend(),
            "zone_distribution": {
                "green": sum(1 for z in self.zone_history() if z == ConfidenceZone.GREEN),
                "yellow": sum(1 for z in self.zone_history() if z == ConfidenceZone.YELLOW),
                "red": sum(1 for z in self.zone_history() if z == ConfidenceZone.RED),
            }
        }
