"""Gate G5 — power restatement for the frozen main grid.

Computes, from the pilot/G1/G2b effect sizes, the per-cell n the
primary contrasts need at alpha=0.05 (two-sided), power=0.80, and
checks them against what the frozen design delivers: 40 chains
(G4) x 5 fault conditions x 3 seeds, chain = independent unit
(amendment section F.16), models always separate arms.

Contrasts:
  1. Formation M1 vs M3 (paired by origin, McNemar) — pilot
     Qwen2.5-7B M1 0.700 vs M3-nonempty 0.500; discordant-pair
     split assumed p10=0.30 / p01=0.10 (consistent with both
     marginals and the pilot's paired table direction).
  2. Propagation (H2 headline): eligible-subset M1 downstream
     incorrect vs the M5 floor, plus CI half-width for the
     propagation rate itself. Eligibility sensitivity 0.6-0.8
     (pilot origin success was 38/60; the G2b smoke's floor held
     everywhere, so this brackets the plausible range).
  3. M3 evidence-visibility dissociation (pilot 0.083 visible vs
     0.750 invisible) — the per-fault mechanism claim.
  4. Persistence / correction resistance: conditional on
     propagation events; reported as CI widths, no power claim.
"""
from __future__ import annotations

import math

Z_A = 1.959964  # alpha 0.05 two-sided
Z_B = 0.841621  # power 0.80
CHAINS = 40
FAULTS = 5
SEEDS = 3


def n_two_prop(p1: float, p2: float) -> float:
    """Per-arm n, normal approximation, equal arms."""
    num = (Z_A + Z_B) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    return num / (p1 - p2) ** 2


def n_mcnemar(p10: float, p01: float) -> float:
    """Total paired origins needed, normal approximation."""
    return (Z_A + Z_B) ** 2 * (p10 + p01) / (p10 - p01) ** 2


def ci_half_width(p: float, n: float) -> float:
    return Z_A * math.sqrt(p * (1 - p) / n)


def main() -> None:
    origins_per_model = CHAINS * FAULTS * SEEDS
    origins_per_fault = CHAINS * SEEDS
    print(f"Frozen design: {CHAINS} chains x {FAULTS} faults x {SEEDS} seeds")
    print(f"  origins per model (per policy): {origins_per_model}")
    print(f"  origins per model per fault:    {origins_per_fault}")
    print(f"  origin runs per model, all 6 policies: {origins_per_model * 6}")
    print()

    # 1. Formation M1 vs M3, paired.
    need = n_mcnemar(0.30, 0.10)
    print("1. Formation M1 0.700 vs M3-nonempty 0.500 (McNemar, paired by origin)")
    print(f"   origins needed per model (fault-pooled): {math.ceil(need)}")
    print(f"   delivered: {origins_per_model} (fault-pooled), "
          f"{origins_per_fault} per fault")
    d_chain = (Z_A + Z_B) * math.sqrt(0.40 / CHAINS)
    d_cs = (Z_A + Z_B) * math.sqrt(0.40 / origins_per_fault)
    print(f"   detectable paired difference at 40 chains only: {d_chain:.3f}; "
          f"at chain x seed (120): {d_cs:.3f} (effect 0.200)")
    print()

    # Formation rate precision.
    print("   Formation-rate CI half-widths (worst case p=0.5):")
    print(f"   per (model, policy): n={origins_per_model} -> "
          f"+/-{ci_half_width(0.5, origins_per_model):.3f}")
    print(f"   per (model, policy, fault): n={origins_per_fault} -> "
          f"+/-{ci_half_width(0.5, origins_per_fault):.3f}")
    print()

    # 2. Propagation.
    print("2. Propagation (eligible M1 chain-runs per model: "
          f"{CHAINS * 4 * SEEDS} across the 4 non-control faults)")
    for elig in (0.6, 0.7, 0.8):
        n_elig = CHAINS * 4 * SEEDS * elig
        print(f"   eligibility {elig:.1f}: n={n_elig:.0f} eligible; "
              f"propagation CI half-width at p=0.25: "
              f"+/-{ci_half_width(0.25, n_elig):.3f}; "
              f"M1 0.25 vs M5 0.05 needs {math.ceil(n_two_prop(0.25, 0.05))}/arm")
    print()

    # 3. M3 visibility dissociation.
    print("3. M3 dissociation: visible 0.083 vs invisible 0.750")
    print(f"   per-arm n needed: {math.ceil(n_two_prop(0.083, 0.750))} "
          f"(delivered per visibility class per model: "
          f"{CHAINS * 2 * SEEDS} invisible / {CHAINS * 3 * SEEDS} visible origins)")
    print()

    # 4. Conditional outcomes.
    print("4. Persistence / correction resistance (conditional on propagation)")
    for elig in (0.6, 0.8):
        events = CHAINS * 4 * SEEDS * elig * 0.25
        print(f"   eligibility {elig:.1f}, propagation 0.25: ~{events:.0f} events/model; "
              f"CI half-width at p=0.5: +/-{ci_half_width(0.5, events):.3f}")


if __name__ == "__main__":
    main()
