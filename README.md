# Dynamic Pricing Optimization with Deep Q-Learning (DQN)

> Training a DQN agent to optimally manage product pricing based on demand and competitor prices

---

## Overview

Reinforcement Learning project implementing a DQN agent for dynamic pricing optimization.
The agent learns to maximize revenue while accounting for demand elasticity and competitor pricing behavior.

---

## Problem

A seller needs to set the optimal price for a product at each time step.
The price affects demand (higher price → lower demand), and competitors change their prices randomly.
The agent must balance revenue maximization against losing customers to cheaper competitors.

---

## Environment

Custom **Gymnasium** environment built from scratch:

| Parameter | Value |
|---|---|
| State space | `[current_price, demand, competitor_price]` |
| Action space | 3 discrete actions |
| Episode length | 100 steps |
| Reward | `revenue - penalty for price deviation from competitor` |

**Actions:**
- `0` — decrease price by 0.5
- `1` — keep price unchanged
- `2` — increase price by 0.5

**Demand model:**
```
demand = 100 - 10 * price + 5 * (competitor_price - price) + noise
```

---

## Results

After 50,000 training steps the agent consistently learns to:
- Hold price in range **3.0–4.0** when competitor price is 5.0–7.0
- Adapt pricing dynamically as competitor changes price
- Achieve average reward per step: **~200–250**

| Episode | Total Reward | Total Revenue | Avg Reward/Step |
|---|---|---|---|
| 1 | 19,614 | 20,150 | 196.15 |
| 2 | 24,839 | 26,162 | 248.40 |
| 3 | 24,117 | 25,202 | 241.17 |

---

## DQN Configuration

```python
DQN(
    policy="MlpPolicy",
    learning_rate=1e-3,
    buffer_size=50_000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.3,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    target_update_interval=1000,
    total_timesteps=50_000
)
```

---

## Tech Stack

- **Python 3.10+**
- **Gymnasium** — custom RL environment
- **Stable-Baselines3** — DQN implementation
- **NumPy**

---

## Installation

```bash
pip install numpy gymnasium stable-baselines3
```

---

## Usage

```bash
python dynamic_pricing_dqn.py
```

The script will:
1. Create and validate the custom environment
2. Train the DQN agent (50,000 steps)
3. Save the model to `dynamic_pricing_dqn_model`
4. Run 3 test episodes with detailed step-by-step output

---

## Author

**Aitmukhammed Zhaxylykov**
Astana IT University — Big Data Analysis
[GitHub](https://github.com/Aitosh1)
