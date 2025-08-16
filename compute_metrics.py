#!/usr/bin/env python3
"""
compute_metrics.py
==================
Single source of truth for all metrics reported in the CareRoute NEJM-45 evaluation paper.
Computes concordance, coverage, and burden metrics from the case analysis CSV
using canonicalized triage labels. No coverage values are inferred from feature lists.

Usage
-----
$ python3 compute_metrics.py [path/to/csv]

If no path is supplied, uses nejm45_eval.csv

Outputs
-------
1. Triage concordance (3-tier)
2. Under/over-triage rates
3. Confusion matrix
4. Coverage statistics (overall and by tier)
5. Questions-to-decision statistics (overall and by tier)

Author: Jag Kondru
Date: August 2025
"""

import csv
import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Callable
from collections import defaultdict
import re
import hashlib
import os

CSV_DEFAULT = Path(__file__).parent/ 'nejm45_eval.csv'

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _percent(n: int, d: int) -> float:
    """Calculate percentage."""
    return n / d * 100 if d else 0.0

def _wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for proportions (95% by default)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * (((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5) / denom
    lower = max(0, centre - margin)
    upper = min(1, centre + margin)
    return lower, upper

# ---------------------------------------------------------------------------
# Label normalization helpers
# ---------------------------------------------------------------------------

def _normalize_4tier(label: str) -> str:
    """Return canonical 4-tier label for a variety of inputs."""
    if label is None:
        return ""
    s = label.strip().lower().replace("-", " ")
    s = " ".join(s.split())  # collapse whitespace
    if s in {"emergent care", "emergency care", "emergent", "emergency"}:
        return "Emergency Care"
    if s in {"urgent care", "urgent"}:
        return "Urgent Care"
    if s in {"doctor visit", "doctor", "non emergent", "non emergent care", "nonemergent care", "nonemergent"}:
        return "Doctor Visit"
    if s in {"self care", "self"}:
        return "Self Care"
    # Common dataset phrasings
    if s.startswith("requires emergent"):
        return "Emergency Care"
    if s.startswith("requires non emergent"):
        return "Doctor Visit"
    if s.startswith("self care") or s.startswith("self"):
        return "Self Care"
    return label.strip()

def _normalize_3tier(label: str) -> str:
    """Collapse to 3-tier tokens: Emergency / Doctor / Self."""
    canon = _normalize_4tier(label)
    if canon == "Urgent Care":
        return "Doctor"
    if canon == "Emergency Care":
        return "Emergency"
    if canon == "Doctor Visit":
        return "Doctor"
    if canon == "Self Care":
        return "Self"
    # Fallback best-effort
    low = (label or "").lower()
    if "emerg" in low:
        return "Emergency"
    if "urgent" in low or "doctor" in low or "non" in low:
        return "Doctor"
    return "Self"


# ---------------------------------------------------------------------------
# Feature parsing & coverage computation (new format)
# ---------------------------------------------------------------------------

def _int(value: str) -> int:
    """Parse an integer safely from CSV values."""
    try:
        return int(str(value).strip())
    except Exception:
        return 0


def _split_features(value: str):
    """Split a comma/semicolon-separated feature list into unique, trimmed items."""
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
    # de-duplicate preserving order
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def compute_row_coverage(row: Dict[str, str], numerator_mode: str = 'elicited') -> float:
    """Compute coverage percent (0–100) strictly from CSV counts.

    Numerator modes (counts only):
      - 'elicited' (default): NumFeaturesElicited
      - 'elicited_plus_volunteered': NumFeaturesElicited + NumFeaturesVolunteered
      - 'volunteered': NumFeaturesVolunteered
      - 'missed': NumFeaturesMissed

    Denominator:
      - Always uses NumFeatures (count from CSV). If missing or zero, coverage = 0.
    """
    denom = _int(row.get('NumFeatures', ''))

    if numerator_mode == 'elicited_plus_volunteered':
        num = _int(row.get('NumFeaturesElicited', '')) + _int(row.get('NumFeaturesVolunteered', ''))
    elif numerator_mode == 'volunteered':
        num = _int(row.get('NumFeaturesVolunteered', ''))
    elif numerator_mode == 'missed':
        num = _int(row.get('NumFeaturesMissed', ''))
    else:
        # 'elicited'
        num = _int(row.get('NumFeaturesElicited', ''))

    return _percent(num, denom) if denom else 0.0

# Convenience wrappers for manuscript and optional metrics
def compute_row_coverage_elicited(row: Dict[str, str]) -> float:
    """Primary manuscript metric: NumFeaturesElicited / NumFeatures (counts only)."""
    return compute_row_coverage(row, numerator_mode='elicited')

def compute_row_coverage_volunteered(row: Dict[str, str]) -> float:
    """Optional: NumFeaturesVolunteered / NumFeatures (counts only)."""
    return compute_row_coverage(row, numerator_mode='volunteered')

def compute_row_coverage_missed(row: Dict[str, str]) -> float:
    """Optional: NumFeaturesMissed / NumFeatures (counts only)."""
    return compute_row_coverage(row, numerator_mode='missed')

def compute_row_elicited_as_pct_surfaced(row: Dict[str, str]) -> float:
    """New metric: Elicited as percent of total surfaced features.
    
    Returns: NumFeaturesElicited / (NumFeaturesElicited + NumFeaturesVolunteered) * 100
    
    This metric shows what percentage of the total surfaced features (elicited + volunteered)
    were obtained through elicitation rather than initial volunteering.
    Returns 0.0 if no features were surfaced.
    """
    elicited = _int(row.get('NumFeaturesElicited', ''))
    volunteered = _int(row.get('NumFeaturesVolunteered', ''))
    total_surfaced = elicited + volunteered
    
    return _percent(elicited, total_surfaced) if total_surfaced else 0.0

def map_to_3tier(actual: str, expected: str) -> Tuple[str, str]:
    """Map Actual/Expected strings to 3-tier tokens using canonicalization."""
    exp_cat = _normalize_3tier(expected)
    act_cat = _normalize_3tier(actual)
    return exp_cat, act_cat

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(csv_path: Path) -> List[Dict[str, str]]:
    """Load CSV data."""
    with csv_path.open(encoding='utf-8', newline='') as f:
        # Use csv.DictReader with proper quoting to handle commas within quoted fields
        return list(csv.DictReader(f, quoting=csv.QUOTE_MINIMAL))

# ---------------------------------------------------------------------------
# Concordance metrics
# ---------------------------------------------------------------------------

def compute_concordance(rows: List[Dict[str, str]]) -> Dict:
    """Compute 3-tier concordance, under/over-triage, and confusion matrix."""
    total = len(rows)

    # 3-tier exact matches
    exact = 0

    # Optional: count "justifiable_mismatch" if present in CSV for adjusted reporting
    justified = sum(1 for r in rows if r.get('CareRecommendationScore', '').strip() == 'justifiable_mismatch')

    confusion_matrix = defaultdict(int)
    under_triage = 0
    over_triage = 0
    emergency_under = 0
    under_triage_cases = []
    over_triage_cases = []
    expected_counts = defaultdict(int)

    severity_map = {'Emergency': 3, 'Doctor': 2, 'Self': 1}

    for row in rows:
        exp, act = map_to_3tier(row.get('ActualCareRecommendation', ''), row.get('ExpectedCareRecommendation', ''))
        expected_counts[exp] += 1
        confusion_matrix[(exp, act)] += 1
        if exp == act:
            exact += 1
        if severity_map[act] < severity_map[exp]:
            under_triage += 1
            under_triage_cases.append((row.get('Title') or row.get('CaseTitle') or row.get('Case') or row.get('Vignette') or ''))
            if exp == 'Emergency':
                emergency_under += 1
        elif severity_map[act] > severity_map[exp]:
            over_triage += 1
            over_triage_cases.append((row.get('Title') or row.get('CaseTitle') or row.get('Case') or row.get('Vignette') or ''))

    ci_exact = _wilson_ci(exact, total)
    # Safety on Emergency vignettes: proportion without under-triage among Emergency cases (computed from data)
    emergency_total = expected_counts.get('Emergency', 0)
    safe_emerg_successes = max(0, emergency_total - emergency_under)
    ci_emergency = _wilson_ci(safe_emerg_successes, emergency_total) if emergency_total else (0.0, 1.0)

    return {
        'total': total,
        'exact': exact,
        'justified': justified,
        'mismatch': total - exact - justified if total is not None else 0,
        'exact_pct': _percent(exact, total),
        'adjusted_pct': _percent(exact + justified, total),
        'ci_exact': ci_exact,
        'under_triage': under_triage,
        'under_triage_pct': _percent(under_triage, total),
        'under_triage_cases': under_triage_cases,
        'over_triage': over_triage,
        'over_triage_pct': _percent(over_triage, total),
        'over_triage_cases': over_triage_cases,
        'emergency_under': emergency_under,
        'emergency_total': emergency_total,
        'ci_emergency': ci_emergency,
        'confusion_matrix': dict(confusion_matrix),
        'expected_counts': dict(expected_counts),
    }

# ---------------------------------------------------------------------------
# Efficiency and coverage metrics
# ---------------------------------------------------------------------------

def compute_efficiency_by_tier(rows: List[Dict[str, str]], coverage_fn: Callable[[Dict[str, str]], float]) -> Dict:
    """Compute questions and coverage statistics by 3-tier expected tier."""
    tiers = {
        'Emergency': {'questions': [], 'coverage': []},
        'Doctor': {'questions': [], 'coverage': []},
        'Self': {'questions': [], 'coverage': []}
    }

    for row in rows:
        exp_tier = _normalize_3tier(row.get('ExpectedCareRecommendation', ''))
        tiers.setdefault(exp_tier, {'questions': [], 'coverage': []})
        # Questions
        try:
            tiers[exp_tier]['questions'].append(int(float(str(row.get('NumQuestions', 0)).strip())))
        except Exception:
            pass
        # Coverage from CSV counts via injected function (no list-size fallbacks)
        cov = coverage_fn(row)
        # Always include coverage, even if zero; this avoids biasing medians upward
        tiers[exp_tier]['coverage'].append(cov)

    def _quartiles(vals: List[float]) -> Tuple[float, float]:
        # Inclusive method is more stable for small N and matches many biomedical reports
        if len(vals) >= 4:
            qs = statistics.quantiles(vals, n=4, method='inclusive')
            return qs[0], qs[2]
        else:
            s = sorted(vals)
            return (s[0], s[-1]) if s else (0.0, 0.0)

    results = {}
    for tier, vals in tiers.items():
        q = vals['questions']
        c = vals['coverage']
        if q:
            q1, q3 = _quartiles(q)
            results[tier] = {
                'questions_median': statistics.median(q),
                'questions_q1': q1,
                'questions_q3': q3,
                'questions_total': len(q),
            }
        if c:
            c_q1, c_q3 = _quartiles(c)
            results.setdefault(tier, {})
            results[tier].update({
                'coverage_median': statistics.median(c),
                'coverage_q1': c_q1,
                'coverage_q3': c_q3
            })
    return results

# ---------------------------------------------------------------------------
# Helpers for overall medians
# ---------------------------------------------------------------------------

def _overall_questions(rows: List[Dict[str, str]]) -> Tuple[float, float, float]:
    q = []
    for r in rows:
        try:
            q.append(int(float(str(r.get('NumQuestions', 0)).strip())))
        except Exception:
            pass
    if not q:
        return 0.0, 0.0, 0.0
    if len(q) >= 4:
        qs = statistics.quantiles(q, n=4, method='inclusive')
        q1, q3 = qs[0], qs[2]
    else:
        s = sorted(q)
        q1, q3 = s[0], s[-1]
    return statistics.median(q), q1, q3


def _overall_coverage(rows: List[Dict[str, str]], coverage_fn: Callable[[Dict[str, str]], float]) -> Tuple[float, float, float]:
    """Overall coverage (median and IQR) computed strictly from CSV counts per row."""
    cov = [coverage_fn(r) for r in rows]
    if not cov:
        return 0.0, 0.0, 0.0
    if len(cov) >= 4:
        qs = statistics.quantiles(cov, n=4, method='inclusive')
        q1, q3 = qs[0], qs[2]
    else:
        s = sorted(cov)
        q1, q3 = s[0], s[-1]
    return statistics.median(cov), q1, q3

# ---------------------------------------------------------------------------
# Manuscript output formatter
# ---------------------------------------------------------------------------

def print_manuscript_numbers(rows: List[Dict[str, str]], coverage_fn: Callable[[Dict[str, str]], float]):
    """Print all numbers formatted for direct inclusion in the manuscript."""
    conc = compute_concordance(rows)
    eff = compute_efficiency_by_tier(rows, coverage_fn)

    cov_med, cov_q1, cov_q3 = _overall_coverage(rows, coverage_fn)
    elicit_frac_med, elicit_frac_q1, elicit_frac_q3 = _overall_coverage(rows, compute_row_elicited_as_pct_surfaced)
    q_med, q_q1, q_q3 = _overall_questions(rows)

    print("\n" + "="*80)
    print("MANUSCRIPT NUMBERS")
    print("="*80)

    print("\n## ABSTRACT & RESULTS:")
    print(f"Exact 3-tier concordance: {conc['exact']}/{conc['total']} ({conc['exact_pct']:.1f}%; 95% CI {conc['ci_exact'][0]*100:.1f}-{conc['ci_exact'][1]*100:.1f}%)")
    print(f"Elicitation coverage (overall): median {cov_med:.0f}% (IQR {cov_q1:.0f}-{cov_q3:.0f})")
    print(f"Elicitation fraction (overall): median {elicit_frac_med:.0f}% (IQR {elicit_frac_q1:.0f}-{elicit_frac_q3:.0f})")
    print(f"Questions asked (overall): median {q_med:.0f} (IQR {q_q1:.0f}-{q_q3:.0f})")

    print(f"Under-triage (all tiers): {conc['under_triage']}/{conc['total']} ({conc['under_triage_pct']:.1f}%)")
    if conc['under_triage_cases']:
        print(f"  Under-triage cases: {', '.join(conc['under_triage_cases'])}")
    print(f"Over-triage (all tiers): {conc['over_triage']}/{conc['total']} ({conc['over_triage_pct']:.1f}%)")

    print("\n## CONFUSION MATRIX:")
    print("| NEJM-45 Reference | Emergency | Doctor Visit | Self Care |")
    print("|:---|:---:|:---:|:---:|")
    m = conc['confusion_matrix']
    ec = conc['expected_counts']
    print(f"| Emergency (n={ec.get('Emergency',0)}) | {m.get(('Emergency','Emergency'),0)} | {m.get(('Emergency','Doctor'),0)} | {m.get(('Emergency','Self'),0)} |")
    print(f"| Doctor Visit (n={ec.get('Doctor',0)}) | {m.get(('Doctor','Emergency'),0)} | {m.get(('Doctor','Doctor'),0)} | {m.get(('Doctor','Self'),0)} |")
    print(f"| Self Care (n={ec.get('Self',0)}) | {m.get(('Self','Emergency'),0)} | {m.get(('Self','Doctor'),0)} | {m.get(('Self','Self'),0)} |")

    print("\n## USER-BURDEN & ELICITATION COVERAGE BY TIER:")
    for tier in ['Emergency', 'Doctor', 'Self']:
        if tier in eff and eff[tier]:
            e = eff[tier]
            tier_name = 'Doctor Visit' if tier == 'Doctor' else f'{tier} Care'
            if 'questions_median' in e:
                print(f"{tier_name}: median {e['questions_median']:.0f} questions (IQR {e['questions_q1']:.0f}-{e['questions_q3']:.0f})")
            if 'coverage_median' in e:
                print(f"{tier_name} elicitation coverage: median {e['coverage_median']:.0f}% (IQR {e['coverage_q1']:.0f}-{e['coverage_q3']:.0f})")

    print("\n## SAFETY ON EMERGENCY VIGNETTES:")
    emerg_total = conc.get('emergency_total', 0)
    safe = max(0, emerg_total - conc['emergency_under'])
    ci_lo, ci_hi = conc['ci_emergency']
    if emerg_total:
        print(f"Emergency cases without under-triage: {safe}/{emerg_total} (95% CI {ci_lo*100:.1f}-{ci_hi*100:.1f}%)")
    else:
        print("Emergency cases without under-triage: N/A (no Emergency cases present)")

    print("\n" + "="*80)

# ---------------------------------------------------------------------------
# Optional: Coverage breakdown for volunteered and missed
# ---------------------------------------------------------------------------

def print_coverage_breakdown(rows: List[Dict[str, str]]):
    """Optional: print overall volunteered and missed coverage medians for reference."""
    v_med, v_q1, v_q3 = _overall_coverage(rows, compute_row_coverage_volunteered)
    m_med, m_q1, m_q3 = _overall_coverage(rows, compute_row_coverage_missed)
    e_surf_med, e_surf_q1, e_surf_q3 = _overall_coverage(rows, compute_row_elicited_as_pct_surfaced)
    print("\n## COVERAGE BREAKDOWN (Optional):")
    print(f"Volunteered coverage (overall): median {v_med:.0f}% (IQR {v_q1:.0f}-{v_q3:.0f})")
    print(f"Missed coverage (overall): median {m_med:.0f}% (IQR {m_q1:.0f}-{m_q3:.0f})")
    print(f"Elicitation Fraction: median {e_surf_med:.0f}% (IQR {e_surf_q1:.0f}-{e_surf_q3:.0f})")

# ---------------------------------------------------------------------------
# Validation & diagnostics
# ---------------------------------------------------------------------------

def _sha1_of_file(path: Path) -> str:
    try:
        h = hashlib.sha1()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""

def _try_int(val) -> Tuple[bool, int]:
    try:
        return True, int(float(str(val).strip()))
    except Exception:
        return False, 0

def _describe_distribution(vals: List[float]) -> Tuple[float, float, float, float, float]:
    """Return (min, q1, median, q3, max) using inclusive quartiles."""
    if not vals:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mn = min(vals)
    mx = max(vals)
    if len(vals) >= 4:
        qs = statistics.quantiles(vals, n=4, method='inclusive')
        q1, q3 = qs[0], qs[2]
    else:
        s = sorted(vals)
        q1, q3 = s[0], s[-1]
    med = statistics.median(vals)
    return mn, q1, med, q3, mx

def validate_data(rows: List[Dict[str, str]], csv_path: Path, coverage_fn: Callable[[Dict[str, str]], float]) -> None:
    print("\n" + "="*80)
    print("VALIDATION & DIAGNOSTICS")
    print("="*80)
    try:
        size = os.path.getsize(csv_path)
    except Exception:
        size = -1
    print(f"CSV: {csv_path} | size: {size} bytes | sha1: {_sha1_of_file(csv_path)}")

    # Check numeric fields
    fields = ['NumFeatures','NumFeaturesElicited','NumFeaturesVolunteered','NumFeaturesMissed','NumQuestions']
    bad_numeric = []
    zero_denoms = []
    sum_exceeds = []

    tier_vals = {'Emergency': [], 'Doctor': [], 'Self': []}
    cov_all = []
    q_all = []

    label_issues = []
    for i, r in enumerate(rows, start=1):
        # Label sanity
        exp = _normalize_3tier(r.get('ExpectedCareRecommendation',''))
        if exp not in ('Emergency','Doctor','Self'):
            label_issues.append((i, r.get('ExpectedCareRecommendation','')))
        # Numeric checks
        parsed = {}
        for f in fields:
            ok, val = _try_int(r.get(f, ""))
            if not ok:
                bad_numeric.append((i, f, r.get(f, "")))
            parsed[f] = val

        denom = parsed['NumFeatures']
        num_e = parsed['NumFeaturesElicited']
        num_v = parsed['NumFeaturesVolunteered']
        num_m = parsed['NumFeaturesMissed']

        if denom <= 0 and (num_e > 0 or num_v > 0 or num_m > 0):
            zero_denoms.append((i, denom, num_e, num_v, num_m))
        if (num_e + num_v + num_m) > denom and denom > 0:
            sum_exceeds.append((i, denom, num_e, num_v, num_m))

        # Coverage & questions collections
        cov = coverage_fn(r)
        cov_all.append(cov)
        q_all.append(parsed['NumQuestions'])
        tier_vals.setdefault(exp, []).append(cov)

    # Report issues
    if label_issues:
        print(f"\n[WARN] Unexpected ExpectedCareRecommendation labels in {len(label_issues)} rows (showing up to 5):")
        for t in label_issues[:5]:
            print(f"  Row {t[0]}: '{t[1]}' -> normalized '{_normalize_3tier(t[1])}'")

    if bad_numeric:
        print(f"\n[WARN] Non-numeric cells coerced to 0 in {len(bad_numeric)} places (showing up to 8):")
        for i, f, raw in bad_numeric[:8]:
            print(f"  Row {i} field {f} = '{raw}'")

    if zero_denoms:
        print(f"\n[WARN] Denominator (NumFeatures) is 0 while numerators > 0 in {len(zero_denoms)} rows (showing up to 5):")
        for tup in zero_denoms[:5]:
            i, d, e, v, m = tup
            print(f"  Row {i}: NumFeatures={d}, E/V/M={e}/{v}/{m}")

    if sum_exceeds:
        print(f"\n[WARN] Elicited+Volunteered+Missed exceeds NumFeatures in {len(sum_exceeds)} rows (showing up to 5):")
        for tup in sum_exceeds[:5]:
            i, d, e, v, m = tup
            print(f"  Row {i}: NumFeatures={d}, E/V/M={e}/{v}/{m} (sum={e+v+m})")

    # Overall distributions
    print("\nCoverage % (overall, counts-only):")
    mn, q1, med, q3, mx = _describe_distribution(cov_all)
    print(f"  min={mn:.1f}  Q1={q1:.1f}  median={med:.1f}  Q3={q3:.1f}  max={mx:.1f}")

    print("\nQuestions (overall):")
    mn, q1, med, q3, mx = _describe_distribution(q_all)
    print(f"  min={mn:.0f}  Q1={q1:.0f}  median={med:.0f}  Q3={q3:.0f}  max={mx:.0f}")

    # By-tier distributions
    for tier in ['Emergency','Doctor','Self']:
        vals = tier_vals.get(tier, [])
        if not vals:
            continue
        mn, q1, med, q3, mx = _describe_distribution(vals)
        label = 'Doctor Visit' if tier=='Doctor' else (f'{tier} Care' if tier!='Doctor' else tier)
        print(f"\nCoverage % by tier — {label}:")
        print(f"  min={mn:.1f}  Q1={q1:.1f}  median={med:.1f}  Q3={q3:.1f}  max={mx:.1f}")

    # Show extreme rows (to help locate Q1/Q3 boundaries)
    idx_sorted_cov = sorted(range(len(cov_all)), key=lambda k: cov_all[k])
    if cov_all:
        lo_idxs = idx_sorted_cov[:5]
        hi_idxs = idx_sorted_cov[-5:]
        print("\nLowest 5 coverage rows (row index starting at 1):")
        for li in lo_idxs:
            r = rows[li]
            title = r.get('Title') or r.get('Case') or r.get('Vignette') or ''
            print(f"  Row {li+1}: {title[:80]}...  coverage={cov_all[li]:.1f}%  NumFeaturesElicited={_int(r.get('NumFeaturesElicited',''))} / NumFeatures={_int(r.get('NumFeatures',''))}")
        print("\nHighest 5 coverage rows:")
        for hi in hi_idxs:
            r = rows[hi]
            title = r.get('Title') or r.get('Case') or r.get('Vignette') or ''
            print(f"  Row {hi+1}: {title[:80]}...  coverage={cov_all[hi]:.1f}%  NumFeaturesElicited={_int(r.get('NumFeaturesElicited',''))} / NumFeatures={_int(r.get('NumFeatures',''))}")

    print("\n" + "="*80)

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute manuscript metrics for NEJM-45 evaluation.')
    parser.add_argument('csv_path', nargs='?', default=str(CSV_DEFAULT), help='Path to the evaluation CSV (default: data/nejm45_eval.csv)')
    parser.add_argument('--show-coverage-breakdown', action='store_true',
                        help='Also print volunteered and missed coverage medians (overall).')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation checks and diagnostics.')
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        sys.exit(f"Error: CSV file not found at {csv_path}")

    rows = load_data(csv_path)
    print(f"Loaded {len(rows)} cases from {csv_path}")

    coverage_fn = compute_row_coverage_elicited

    # Print manuscript-ready numbers (strict elicitation coverage)
    print_manuscript_numbers(rows, coverage_fn)

    if args.show_coverage_breakdown:
        print_coverage_breakdown(rows)

    if args.validate:
        validate_data(rows, csv_path, coverage_fn)