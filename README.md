# Universal Density-Based Optimization Algorithm 

## Overview
introduces a universal, density-driven optimization algorithm based on a **molecular fusion mechanism** and **adaptive micro line search**.  
The algorithm requires no problem-specific tuning and performs consistently across smooth, multimodal, and deceptive landscapes.



---

## Core Principles
- The algorithm moves toward **regions of higher population density**, where stable minima naturally form.
- A **neutral fusion law** combines candidate solutions per-dimension using stable softmax weighting.
- A **5‑point adaptive micro line search** refines each coordinate locally.
- A **self-regulating temperature schedule** controls exploration vs. exploitation.
- No heuristics, patches, or conditional rules are used.

---

## Key Features
- **Density-driven exploration**  
- **Molecular fusion per dimension**  
- **Stable softmax weighting**  
- **Adaptive microsearch**  
- **Nonlinear noise regulation**  
- **Universal behavior across landscapes**

---

## Benchmark Results
The algorithm has been tested on standard 10‑dimensional benchmarks:

| Function | Result |
|---------|--------|
| Sphere | ~0 |
| Rastrigin | ~0 |
| Ackley | ~0 |
| Griewank | ~0 |
| Schwefel | ~0.1 (global minimum near 421) |

These results were obtained without parameter tuning.

---


## 📦 Installation

The optimizer only requires NumPy:

```bash
pip install numpy

Support the Project & Voluntary Payment (Donation-ware)
The EHSAN Algorithm is released as free and open-source software (under the MIT License). You are free to use it for any purpose, commercial or non-commercial.

However, the creation and maintenance of this unique, research-grade algorithm required significant time and effort. If the EHSAN Algorithm has helped you solve a complex optimization problem, saved you development time, or if you simply value the scientific contribution of this physics-inspired framework:

We strongly encourage you to support the developer's time by making a voluntary donation of any amount you feel is appropriate.

Your support directly funds future research, ongoing maintenance, and the development of new features.

## 💖 Support the Project & Voluntary Payment

The EHSAN Algorithm is released as free and open-source software (under the MIT License). You are free to use it for any purpose.

If EHSAN has provided value for your work, **we strongly encourage you to support the developer's time by making a voluntary donation.** Your funds will directly support continued research and maintenance.

### I. For International Supporters (Global Donations)

Due to international banking restrictions, we only accept cryptocurrency donations from outside Iran. Please ensure you select the **TRON (TRC20)** network when sending USDT.

| Currency | Network | Wallet Address |
| :---: | :---: | :--- |
| **Tether (USDT)** | **TRON (TRC20)** | **`THRNhb27Znc2P4UnrAHZMrP3Cjxdrt7M5c`** |


