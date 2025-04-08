# Adaptive Expert Scaling in SparseMoE

```
Before Adaptation                     After Adaptation
+---------------------+              +---------------------+
| Expert 1 (128 dim)  | <----- 10%  | Expert 1 (64 dim)   | ← Shrunk (rarely used)
+---------------------+              +---------------------+
                                    
| Expert 2 (128 dim)  | <----- 45%  | Expert 2 (256 dim)  | ← Grew (frequently used)
+---------------------+              +---------------------+
                                    
| Expert 3 (128 dim)  | <----- 20%  | Expert 3 (128 dim)  | ← Unchanged (average use)
+---------------------+              +---------------------+
                                    
| Expert 4 (128 dim)  | <----- 25%  | Expert 4 (192 dim)  | ← Grew (above average use)
+---------------------+              +---------------------+

                                    Total params remain constant
```

## Concept Diagram

```
           ┌─────────────────┐
Input ────►│  Router Network │
           └────────┬────────┘
                    │
                    ▼
          ┌─────────────────┐
          │  Expert Usage   │
          │    Tracker      │
          └────────┬────────┘
                   │
                   │   Periodic analysis
                   ▼
       ┌─────────────────────────┐
       │  Expert Scaling Decision │
       └─────────┬───────────────┘
                 │
 ┌───────────────┼──────────────┐
 │               │              │
 ▼               ▼              ▼
┌────────┐   ┌────────┐   ┌────────┐
│Expert 1│   │Expert 2│   │Expert 3│  ... and so on
│ Resize │   │ Resize │   │ Resize │
└────────┘   └────────┘   └────────┘
```

## Implementation Overview

The adaptive expert scaling mechanism consists of several key components:

1. **ExpertUsageTracker**: Monitors which experts are used and how important they are
2. **AdaptiveExpert**: Expert implementation that can dynamically resize its internal representation
3. **AdaptiveExpertLayer**: Layer that manages multiple adaptive experts and coordinates their scaling

### Dynamic Sizing Mechanism

1. For each forward pass, we track:
   - Which experts were selected
   - The importance/routing weight of each expert
   
2. We maintain an exponential moving average (EMA) of expert importance

3. When the scaling interval is reached:
   - Compute scaling factors based on importance EMA
   - Calculate new dimensions for each expert
   - Rescale expert internal weights while preserving important connections

4. The system carefully balances parameter allocation to maintain:
   - Consistent total parameter count
   - Appropriate minimum and maximum expert sizes
   - Smooth transitions during resizing

### Visualization Tools

SparseMoE includes built-in visualization tools to track expert scaling:

1. **Size Evolution Charts**: Track how expert dimensions change over time
2. **Importance Distribution**: View the relative importance of each expert
3. **Utilization Heatmaps**: See which experts are activated for different inputs

These visualizations help understand how the model adapts to different data patterns.