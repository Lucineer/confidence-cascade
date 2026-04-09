# Confidence Cascade

Three-zone confidence propagation (GREEN/YELLOW/RED) for multi-signal decision systems.

## Zones

| Zone | Range | Action |
|------|-------|--------|
| GREEN | 0.85-1.0 | Auto-proceed |
| YELLOW | 0.60-0.85 | Proceed with caution |
| RED | 0.00-0.60 | Stop and investigate |

## Usage

```python
from cascade import create_confidence, sequential_cascade, parallel_cascade, ConfidenceTracker

c1 = create_confidence(0.95, 'ml_model', 0.5)
c2 = create_confidence(0.70, 'rules', 0.3)
result, zones = parallel_cascade([
    {'confidence': c1, 'weight': 0.5},
    {'confidence': c2, 'weight': 0.3}])

tracker = ConfidenceTracker('model')
tracker.record(0.92)
print(tracker.stats())
```

Part of the [Lucineer ecosystem](https://github.com/Lucineer/the-fleet).