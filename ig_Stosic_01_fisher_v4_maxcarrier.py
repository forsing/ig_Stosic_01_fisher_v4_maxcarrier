from __future__ import annotations

# IG = Information Geometry (informaciona geometrija) 

"""
inspiration / upgrade  <--->  inspiracija / nadogradnja


Dragan Stošić / dva rada LUCES / ESP32 osvetljenje: 

1. Empirijska IG: Fisher metric, Multi-Chart (kad signal padne prelaz chartova), Christoffel / Levi-Civita, Histerezis.
https://zenodo.org/records/20094759
(DOI 10.5281/zenodo.20094759) — Fisher, chartovi, Christoffel, histerezis.

2. Ceo experimentalni sloj (paper + data + PVS) — ovo je „journal-ready“ paket. 
isti Manifold + mikro-ekscitacija + Fisher-preconditioned kontrola (A/B −25% jitter) + PVS dokazi + senzorski CSV.
https://zenodo.org/records/20389804
(novija PDF verzija: https://zenodo.org/records/20393695)
Naslov: Excitation-Dependent Observability Geometry…
Sadrži: paper 15 str, 6 CSV (boot…), serial logovi, PVS dokazi, A/B Boot 291 (GEO −25% jitter).
"""


"""
Fisher metrika na porodici raspodela nad istorijom (npr. frekvencije / uslovne raspodele)
multi-chart kad „observabilnost“ padne (npr. drugačiji režim / era)
natural gradient (Fisher precondition) ako nešto optimizujem 
histerezis putanja kroz vreme
mikro-ekscitacija (loto ne možeš da „probudiš“ kao lampu); PVS dokazi.
"""



"""
v4: max umesto prosek
Za y: rate_y = max_{x∈last} P(y|x) — jači signal jednog nosača, ne razvodnjen prosek. Isti Fisher + next.

max P(y|x) → Fisher → next.


IG korak 1 v4 — isto kao v3, ali max umesto proseka.

Za y: rate_y = max_{x ∈ last} P(y|x ∈ draw_t)
pa simplex → Fisher g_ii=1/p_i
Skor: (p_cond − p_global)·√g
next: jedna kombinacija. CSV ceo, seed=39.
"""



import csv
from collections import Counter
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4650_k56.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def global_p(draws: np.ndarray) -> np.ndarray:
    cnt = Counter(draws.reshape(-1).tolist())
    n_slots = len(draws) * FRONT_SELECT
    return np.array([cnt.get(i, 0) / n_slots for i in range(1, FRONT_N + 1)], dtype=float)


def carrier_transition_rates(draws: np.ndarray) -> np.ndarray:
    """M[x-1, y-1] = P(y ∈ next | x ∈ draw_t)"""
    present = np.zeros((len(draws), FRONT_N), dtype=np.float64)
    for i, d in enumerate(draws):
        for x in d.tolist():
            present[i, int(x) - 1] = 1.0

    co = np.zeros((FRONT_N, FRONT_N), dtype=np.float64)
    cnt_x = np.zeros(FRONT_N, dtype=np.float64)
    for t in range(len(draws) - 1):
        xs = np.where(present[t] == 1.0)[0]
        ys = np.where(present[t + 1] == 1.0)[0]
        for xi in xs:
            cnt_x[xi] += 1.0
            for yi in ys:
                co[xi, yi] += 1.0
    return co / np.clip(cnt_x[:, None], 1.0, None)


def conditional_from_last_max(rates: np.ndarray, last: np.ndarray) -> np.ndarray:
    """max P(y|x) preko x u last → Laplace + simplex."""
    carriers = [int(x) - 1 for x in last.tolist()]
    mass = rates[carriers].max(axis=0)
    mass = mass + 1e-6
    return mass / mass.sum()


def fisher_diagonal(p: np.ndarray) -> np.ndarray:
    return 1.0 / np.clip(p, 1e-18, None)


def number_scores(p_cond: np.ndarray, p_glob: np.ndarray, g: np.ndarray) -> dict[int, float]:
    return {
        i + 1: float((p_cond[i] - p_glob[i]) * np.sqrt(g[i]))
        for i in range(FRONT_N)
    }


def _combo_fit(
    combo: list[int],
    score: dict[int, float],
    target_sum: float,
    pos_means: list[float],
    target_odd: float,
) -> float:
    nums = sorted(combo)
    s = sum(score[x] for x in nums)
    s -= 0.08 * abs(sum(nums) - target_sum)
    s -= 0.04 * sum(abs(nums[i] - pos_means[i]) for i in range(FRONT_SELECT))
    odd = sum(1 for x in nums if x % 2)
    s -= 0.3 * abs(odd - target_odd)
    return s


def predict_next(
    draws: np.ndarray,
    p_cond: np.ndarray,
    p_glob: np.ndarray,
    g: np.ndarray,
) -> list[int]:
    score = number_scores(p_cond, p_glob, g)
    ranked = sorted(score, key=lambda n: (-score[n], n))
    target_sum = float(draws.sum(axis=1).mean())
    pos_means = [float(draws[:, i].mean()) for i in range(FRONT_SELECT)]
    target_odd = float(np.mean([sum(1 for x in d if x % 2) for d in draws]))

    candidates = [sorted(ranked[:FRONT_SELECT])]
    for start in range(0, min(20, FRONT_N - FRONT_SELECT + 1)):
        candidates.append(sorted(ranked[start : start + FRONT_SELECT]))

    best, best_fit = None, -1e18
    for base in candidates:
        fit = _combo_fit(base, score, target_sum, pos_means, target_odd)
        if fit > best_fit:
            best_fit, best = fit, list(base)
        for i in range(FRONT_SELECT):
            for repl in ranked[:30]:
                cand = sorted(set(base[:i] + base[i + 1 :] + [repl]))
                if len(cand) != FRONT_SELECT:
                    continue
                fit = _combo_fit(cand, score, target_sum, pos_means, target_odd)
                if fit > best_fit:
                    best_fit, best = fit, cand
    return best if best is not None else sorted(ranked[:FRONT_SELECT])


def run_ig_01_v4(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    last = draws[-1]
    p_glob = global_p(draws)
    rates = carrier_transition_rates(draws)
    p_cond = conditional_from_last_max(rates, last)
    g = fisher_diagonal(p_cond)

    print(f"CSV: {csv_path.name}")
    print(f"Kola: {len(draws)} | seed={SEED} | ig_01_v4 Fisher max P(y|x)")
    print(f"last: {last.tolist()}")
    print()

    anisotropy = float(g.max() / g.min()) if g.min() > 0 else float("inf")
    print("=== p_cond (max P(y|x), x∈last) + Fisher ===")
    print(
        {
            "sum_p": round(float(p_cond.sum()), 6),
            "p_min": float(p_cond.min()),
            "p_max": float(p_cond.max()),
            "g_min": float(g.min()),
            "g_max": float(g.max()),
            "anisotropy": round(anisotropy, 4),
        }
    )
    print()

    score = number_scores(p_cond, p_glob, g)
    ranked = sorted(
        ((n, float(p_cond[n - 1]), float(score[n])) for n in range(1, FRONT_N + 1)),
        key=lambda t: (-t[2], t[0]),
    )
    print("=== top12 po (p_cond − p_glob)·√g ===")
    print([(n, round(pc, 5), round(sc, 5)) for n, pc, sc in ranked[:12]])
    print()

    # koji nosilac daje max za top y
    top_n = ranked[0][0]
    carriers = last.tolist()
    donor = max(carriers, key=lambda x: rates[int(x) - 1, top_n - 1])
    print(
        f"=== donor za top y={top_n}: x={donor}  P={rates[int(donor)-1, top_n-1]:.4f} ==="
    )
    print()

    combo = predict_next(draws, p_cond, p_glob, g)
    print("=== next (ig_01_v4 max nosilac Fisher) ===")
    print("next:", combo)


if __name__ == "__main__":
    run_ig_01_v4()



"""
CSV: loto7_4650_k56.csv
Kola: 4650 | seed=39 | ig_01_v4 Fisher max P(y|x)
last: [4, 5, 6, 11, 12, 18, 28]

=== p_cond (max P(y|x), x∈last) + Fisher ===
{'sum_p': 1.0, 'p_min': 0.021958876637839877, 'p_max': 0.029440169144269755, 'g_min': 33.96719614957241, 'g_max': 45.53967019773609, 'anisotropy': 1.3407}

=== top12 po (p_cond − p_glob)·√g ===
[(12, 0.02654, 0.00922), (23, 0.02944, 0.009), (26, 0.02824, 0.00881), (30, 0.02553, 0.0081), (3, 0.02686, 0.00791), (6, 0.02639,0.00754), (21, 0.02655, 0.00699), (16, 0.02696, 0.00684), (5, 0.02664, 0.00662), (9, 0.02696, 0.0059), (15, 0.02543, 0.00479), (24, 0.02658, 0.00417)]

=== donor za top y=12: x=28  P=0.2039 ===

=== next (ig_01_v4 max nosilac Fisher) ===
next: [5, x, 15, y, 24, z, 36] 
"""
