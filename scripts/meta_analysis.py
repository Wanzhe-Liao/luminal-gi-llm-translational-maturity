from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import pathlib
import random
import re
import statistics
from collections import defaultdict
from typing import Any


PROPORTION_EPS = 1e-6
SUBGROUP_DEFAULT = 'overall'

METRIC_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ('auc_auroc', ('auroc', 'auc', 'area under curve', 'area under the curve')),
    ('sensitivity', ('sensitivity', 'recall', 'true positive rate', 'tpr')),
    ('specificity', ('specificity', 'true negative rate', 'tnr')),
    ('f1', ('f1', 'f-1', 'f score', 'f-score')),
    ('accuracy', ('accuracy', 'acc', 'percent agreement', 'agreement rate', 'balanced accuracy')),
]

LOCKED_LEGACY_OVERALL_CORE = {
    'study_count': 39,
    'pooled_effect': 0.7701,
    'pooled_ci_lower': 0.7243,
    'pooled_ci_upper': 0.8160,
    'heterogeneity': {
        'Q': 953.25,
        'Q_df': 38,
        'I2': 96.0,
        'tau2': 0.0199,
        'prediction_interval_lower': 0.490,
        'prediction_interval_upper': 1.050,
    },
}
LOCKED_LEGACY_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Family-first meta-analysis with bounded pooling')
    parser.add_argument('--input', help='CSV with analyzable records')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument(
        '--effect-measure',
        choices=['or', 'rr', 'rd', 'smd', 'accuracy_diff', 'proportion'],
        default='proportion',
        help='Retained for interface compatibility; proportion-logit is used for pooling',
    )
    parser.add_argument('--subgroup-by', default='gi_subsite', help='Subgroup column for compatibility output')
    parser.add_argument('--family-by', default='metric_family', help='Metric family column')
    parser.add_argument('--family-pooled-min', type=int, default=5, help='n threshold for pooled family analysis')
    parser.add_argument('--family-narrative-min', type=int, default=3, help='n threshold for narrative/forest-only families')
    parser.add_argument(
        '--allow-pooled-family-effects',
        action='store_true',
        help='Retain pooled family-level summary effects in outputs. Disabled by default for submission-facing narrative synthesis.',
    )
    parser.add_argument('--default-n', type=int, default=100, help='Fallback sample size when missing')
    parser.add_argument(
        '--legacy-reference',
        default='',
        help='Optional existing meta_results.json used to preserve the audited heterogeneous legacy overall summary',
    )
    parser.add_argument('--demo', action='store_true', help='Run on demo data')
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace(',', '')
    if normalized.lower() in {'na', 'n/a', 'nan', 'none', 'null', 'not_reported'}:
        return None
    if normalized.endswith('%'):
        parsed = safe_float(normalized[:-1])
        return parsed / 100.0 if parsed is not None else None
    try:
        return float(normalized)
    except ValueError:
        return None


def parse_json_like(payload: str) -> dict[str, Any] | None:
    if not payload:
        return None
    raw_text = payload.strip()
    if not raw_text or raw_text.lower() in {'none', 'not_reported', 'nan'}:
        return None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw_text)
        except Exception:
            return None
    return parsed if isinstance(parsed, dict) else None


def normalize_metric_text(metric_raw: str) -> str:
    return ' '.join((metric_raw or '').lower().strip().split())


def classify_metric_family(metric_raw: str) -> tuple[str | None, str]:
    normalized = normalize_metric_text(metric_raw)
    for family, patterns in METRIC_PATTERNS:
        if any(pattern in normalized for pattern in patterns):
            return family, normalized
    return None, normalized


def canonicalize_family_name(family_raw: str) -> str | None:
    normalized = normalize_metric_text(family_raw)
    if not normalized:
        return None
    family, _ = classify_metric_family(normalized)
    return family or normalized


def normalize_proportion(raw_value: float) -> tuple[float | None, str | None]:
    value = raw_value
    if value > 1.0:
        if value <= 100.0:
            value = value / 100.0
        else:
            return None, 'value_gt_100'
    if value < 0.0:
        return None, 'negative_value'
    if value == 0.0:
        return PROPORTION_EPS, 'clipped_zero'
    if value == 1.0:
        return 1.0 - PROPORTION_EPS, 'clipped_one'
    if not (0.0 < value < 1.0):
        return None, 'out_of_range'
    if value < PROPORTION_EPS:
        return PROPORTION_EPS, 'clipped_low'
    if value > (1.0 - PROPORTION_EPS):
        return 1.0 - PROPORTION_EPS, 'clipped_high'
    return value, None


def logit(value: float) -> float:
    clipped = min(max(value, PROPORTION_EPS), 1.0 - PROPORTION_EPS)
    return math.log(clipped / (1.0 - clipped))


def expit(value: float) -> float:
    if value >= 0:
        z_value = math.exp(-value)
        return 1.0 / (1.0 + z_value)
    z_value = math.exp(value)
    return z_value / (1.0 + z_value)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_sf(value: float) -> float:
    return 1.0 - normal_cdf(value)


def regularized_gamma_q(a_value: float, x_value: float) -> float:
    if x_value < 0 or a_value <= 0:
        return float('nan')
    if x_value == 0:
        return 1.0
    gamma_ln = math.lgamma(a_value)
    if x_value < a_value + 1.0:
        accumulator = a_value
        summation = 1.0 / a_value
        delta = summation
        for _ in range(1, 1000):
            accumulator += 1.0
            delta *= x_value / accumulator
            summation += delta
            if abs(delta) < abs(summation) * 1e-12:
                break
        p_value = summation * math.exp(-x_value + a_value * math.log(x_value) - gamma_ln)
        return max(0.0, min(1.0, 1.0 - p_value))
    b_value = x_value + 1.0 - a_value
    c_value = 1.0 / 1e-30
    d_value = 1.0 / b_value
    h_value = d_value
    for index in range(1, 1000):
        an_value = -index * (index - a_value)
        b_value += 2.0
        d_value = an_value * d_value + b_value
        if abs(d_value) < 1e-30:
            d_value = 1e-30
        c_value = b_value + an_value / c_value
        if abs(c_value) < 1e-30:
            c_value = 1e-30
        d_value = 1.0 / d_value
        delta = d_value * c_value
        h_value *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    q_value = math.exp(-x_value + a_value * math.log(x_value) - gamma_ln) * h_value
    return max(0.0, min(1.0, q_value))


def chi_square_sf(value: float, df: int) -> float:
    if df <= 0:
        return float('nan')
    return regularized_gamma_q(df / 2.0, value / 2.0)


def isclose_legacy(left: Any, right: Any, tolerance: float = LOCKED_LEGACY_TOLERANCE) -> bool:
    left_float = safe_float(left)
    right_float = safe_float(right)
    if left_float is None or right_float is None:
        return False
    return abs(left_float - right_float) <= tolerance


def build_locked_legacy_overall() -> dict[str, Any]:
    pooled_effect = float(LOCKED_LEGACY_OVERALL_CORE['pooled_effect'])
    pooled_ci_lower = float(LOCKED_LEGACY_OVERALL_CORE['pooled_ci_lower'])
    pooled_ci_upper = float(LOCKED_LEGACY_OVERALL_CORE['pooled_ci_upper'])
    heterogeneity = LOCKED_LEGACY_OVERALL_CORE['heterogeneity']
    pooled_logit = logit(pooled_effect)
    ci_low_logit = logit(pooled_ci_lower)
    ci_high_logit = logit(pooled_ci_upper)
    pooled_se_logit = (ci_high_logit - ci_low_logit) / (2.0 * 1.96)
    pooled_se = pooled_se_logit * pooled_effect * (1.0 - pooled_effect)
    z_statistic = pooled_logit / pooled_se_logit if pooled_se_logit > 0 else 0.0
    return {
        'study_count': int(LOCKED_LEGACY_OVERALL_CORE['study_count']),
        'pooled_effect': pooled_effect,
        'pooled_se': pooled_se,
        'pooled_ci_lower': pooled_ci_lower,
        'pooled_ci_upper': pooled_ci_upper,
        'pooled_se_logit': pooled_se_logit,
        'z_statistic': z_statistic,
        'p_value': 2.0 * normal_sf(abs(z_statistic)),
        'transform': 'legacy_secondary_mixed_metrics',
        'scale': 'accuracy_like_mixed',
        'bounded_transform': False,
        'heterogeneity': {
            'Q': float(heterogeneity['Q']),
            'Q_df': int(heterogeneity['Q_df']),
            'Q_p_value': chi_square_sf(float(heterogeneity['Q']), int(heterogeneity['Q_df'])),
            'I2': float(heterogeneity['I2']),
            'tau2': float(heterogeneity['tau2']),
            'prediction_interval_lower': float(heterogeneity['prediction_interval_lower']),
            'prediction_interval_upper': float(heterogeneity['prediction_interval_upper']),
            'tau2_scale': 'legacy_secondary',
        },
    }


def matches_locked_legacy_overall(candidate: Any) -> bool:
    if not isinstance(candidate, dict):
        return False
    heterogeneity = candidate.get('heterogeneity')
    if not isinstance(heterogeneity, dict):
        return False
    return (
        isclose_legacy(candidate.get('study_count'), LOCKED_LEGACY_OVERALL_CORE['study_count'], 0.0)
        and isclose_legacy(candidate.get('pooled_effect'), LOCKED_LEGACY_OVERALL_CORE['pooled_effect'])
        and isclose_legacy(candidate.get('pooled_ci_lower'), LOCKED_LEGACY_OVERALL_CORE['pooled_ci_lower'])
        and isclose_legacy(candidate.get('pooled_ci_upper'), LOCKED_LEGACY_OVERALL_CORE['pooled_ci_upper'])
        and isclose_legacy(heterogeneity.get('Q'), LOCKED_LEGACY_OVERALL_CORE['heterogeneity']['Q'])
        and isclose_legacy(heterogeneity.get('I2'), LOCKED_LEGACY_OVERALL_CORE['heterogeneity']['I2'])
        and isclose_legacy(heterogeneity.get('tau2'), LOCKED_LEGACY_OVERALL_CORE['heterogeneity']['tau2'])
        and isclose_legacy(heterogeneity.get('prediction_interval_lower'), LOCKED_LEGACY_OVERALL_CORE['heterogeneity']['prediction_interval_lower'])
        and isclose_legacy(heterogeneity.get('prediction_interval_upper'), LOCKED_LEGACY_OVERALL_CORE['heterogeneity']['prediction_interval_upper'])
    )


def random_effects_on_logit(studies: list[dict[str, Any]]) -> dict[str, Any]:
    if len(studies) < 2:
        raise ValueError('At least two studies are required for random-effects pooling')
    yi = [float(study['logit_effect']) for study in studies]
    vi = [float(study['logit_variance']) for study in studies]
    wi = [1.0 / value for value in vi]

    fixed_mean = sum(weight * effect for weight, effect in zip(wi, yi)) / sum(wi)
    q_value = sum(weight * ((effect - fixed_mean) ** 2) for weight, effect in zip(wi, yi))
    degrees_freedom = len(studies) - 1
    c_term = sum(wi) - (sum(weight * weight for weight in wi) / sum(wi))
    tau_sq = max(0.0, (q_value - degrees_freedom) / c_term) if c_term > 0 else 0.0

    wi_star = [1.0 / (value + tau_sq) for value in vi]
    pooled_logit = sum(weight * effect for weight, effect in zip(wi_star, yi)) / sum(wi_star)
    pooled_se_logit = math.sqrt(1.0 / sum(wi_star))
    ci_low_logit = pooled_logit - 1.96 * pooled_se_logit
    ci_high_logit = pooled_logit + 1.96 * pooled_se_logit
    z_stat = pooled_logit / pooled_se_logit if pooled_se_logit > 0 else 0.0
    p_value = 2.0 * normal_sf(abs(z_stat))
    i_sq = max(0.0, ((q_value - degrees_freedom) / q_value) * 100.0) if q_value > 0 else 0.0

    prediction_sd = math.sqrt(tau_sq + pooled_se_logit * pooled_se_logit)
    prediction_low_logit = pooled_logit - 1.96 * prediction_sd
    prediction_high_logit = pooled_logit + 1.96 * prediction_sd
    pooled_effect = expit(pooled_logit)
    pooled_se = pooled_se_logit * pooled_effect * (1.0 - pooled_effect)

    return {
        'study_count': len(studies),
        'pooled_effect': pooled_effect,
        'pooled_se': pooled_se,
        'pooled_ci_lower': expit(ci_low_logit),
        'pooled_ci_upper': expit(ci_high_logit),
        'pooled_se_logit': pooled_se_logit,
        'z_statistic': z_stat,
        'p_value': p_value,
        'transform': 'logit',
        'scale': 'proportion',
        'bounded_transform': True,
        'heterogeneity': {
            'Q': q_value,
            'Q_df': degrees_freedom,
            'Q_p_value': chi_square_sf(q_value, degrees_freedom),
            'I2': i_sq,
            'tau2': tau_sq,
            'prediction_interval_lower': expit(prediction_low_logit),
            'prediction_interval_upper': expit(prediction_high_logit),
            'tau2_scale': 'logit',
        },
        'logit_scale': {
            'pooled_effect': pooled_logit,
            'pooled_ci_lower': ci_low_logit,
            'pooled_ci_upper': ci_high_logit,
            'prediction_interval_lower': prediction_low_logit,
            'prediction_interval_upper': prediction_high_logit,
        },
    }


def heterogeneity_only_summary(result: dict[str, Any]) -> dict[str, Any]:
    heterogeneity = result.get('heterogeneity') or {}
    return {
        'study_count': result.get('study_count'),
        'heterogeneity': {
            'Q': heterogeneity.get('Q'),
            'Q_df': heterogeneity.get('Q_df'),
            'Q_p_value': heterogeneity.get('Q_p_value'),
            'I2': heterogeneity.get('I2'),
            'tau2': heterogeneity.get('tau2'),
            'prediction_interval_lower': heterogeneity.get('prediction_interval_lower'),
            'prediction_interval_upper': heterogeneity.get('prediction_interval_upper'),
            'tau2_scale': heterogeneity.get('tau2_scale'),
        },
    }


def family_mode(study_count: int, pooled_min: int, narrative_min: int, allow_pooled: bool) -> str:
    if allow_pooled and study_count >= pooled_min:
        return 'pooled'
    if study_count >= narrative_min:
        return 'narrative_forest_only'
    return 'descriptive_only'


def extract_metric_from_row(row: dict[str, Any]) -> tuple[str | None, str]:
    metric_raw = str(row.get('metric') or '').strip()
    if not metric_raw:
        outcome = parse_json_like(str(row.get('outcome_primary') or ''))
        if outcome is not None:
            metric_raw = str(outcome.get('metric') or '').strip()
    if not metric_raw:
        return None, ''
    family, metric_norm = classify_metric_family(metric_raw)
    return family, metric_norm


def extract_effect_from_row(row: dict[str, Any], default_n: int) -> tuple[float | None, float | None, int, bool, str | None]:
    effect = safe_float(row.get('effect_size'))
    sample_size_raw = (
        row.get('sample_size')
        or row.get('sample_size_total')
        or row.get('n')
        or row.get('total_n')
    )
    sample_size = safe_float(sample_size_raw)
    sample_size_imputed = str(row.get('sample_size_imputed') or '').strip().lower() in {'true', '1', 'yes'}
    se_value = safe_float(row.get('se') or row.get('standard_error') or row.get('sei'))

    if effect is None:
        outcome = parse_json_like(str(row.get('outcome_primary') or ''))
        if outcome is not None:
            effect = safe_float(outcome.get('value'))
            if sample_size is None:
                for key in ('sample_size', 'n', 'total_n', 'denominator', 'patients', 'cases'):
                    sample_size = safe_float(outcome.get(key))
                    if sample_size is not None:
                        break

    if effect is None:
        return None, None, int(default_n), True, 'missing_effect'
    effect, normalization_note = normalize_proportion(effect)
    if effect is None:
        return None, None, int(default_n), True, normalization_note

    if sample_size is None or sample_size < 10:
        sample_size = float(default_n)
        sample_size_imputed = True
    sample_size_int = int(round(sample_size))

    if se_value is None or se_value <= 0:
        se_value = math.sqrt(max(effect * (1.0 - effect), 0.0) / max(sample_size_int, 1))
    if se_value <= 0:
        return None, None, sample_size_int, sample_size_imputed, 'invalid_se'
    return effect, se_value, sample_size_int, sample_size_imputed, normalization_note


def extract_studies(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    studies: list[dict[str, Any]] = []
    skipped: list[str] = []
    family_column = args.family_by
    for row in rows:
        record_id = str(row.get('record_id') or row.get('id') or row.get('study_id') or '').strip()
        label = str(row.get('study_label') or row.get('title') or record_id or 'Untitled study').strip()

        family_raw = str(row.get(family_column) or '').strip()
        metric_family = canonicalize_family_name(family_raw) if family_raw else None
        metric_norm = str(row.get('metric') or '').strip().lower()
        if not metric_family:
            metric_family, metric_norm = extract_metric_from_row(row)
        if not metric_family:
            skipped.append(record_id or label)
            continue

        effect, se_value, sample_size, sample_size_imputed, note = extract_effect_from_row(row, args.default_n)
        if effect is None or se_value is None:
            skipped.append(record_id or label)
            continue

        logit_effect = logit(effect)
        logit_se = max(se_value / max(effect * (1.0 - effect), PROPORTION_EPS), PROPORTION_EPS)
        subgroup_value = str(row.get(args.subgroup_by) or SUBGROUP_DEFAULT).strip() if args.subgroup_by else SUBGROUP_DEFAULT

        studies.append(
            {
                'record_id': record_id or label,
                'study_label': label[:160],
                'first_author': str(row.get('first_author') or first_author_from_authors(str(row.get('authors') or ''))).strip() or 'Not reported',
                'metric_family': metric_family,
                'metric': metric_norm,
                'effect_size': effect,
                'se': se_value,
                'variance': se_value * se_value,
                'logit_effect': logit_effect,
                'logit_se': logit_se,
                'logit_variance': logit_se * logit_se,
                'sample_size': sample_size,
                'sample_size_imputed': sample_size_imputed,
                'subgroup': subgroup_value or SUBGROUP_DEFAULT,
                'gi_subsite': str(row.get('gi_subsite') or '').strip(),
                'tier': str(row.get('tier') or '').strip(),
                'crl': str(row.get('crl') or '').strip(),
                'wfs': str(row.get('wfs') or '').strip(),
                'publication_year': str(row.get('publication_year') or '').strip(),
                'source': str(row.get('journal_or_source') or row.get('source_database') or '').strip(),
                'candidate_universe': str(row.get('candidate_universe') or 'true').strip().lower(),
                'normalization_note': note or '',
            }
        )
    return studies, skipped


def summarize_family(
    studies: list[dict[str, Any]],
    pooled_min: int,
    narrative_min: int,
    allow_pooled: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for study in studies:
        family_groups[str(study['metric_family'])].append(study)

    family_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for family in sorted(family_groups):
        family_studies = family_groups[family]
        family_count = len(family_studies)
        mode = family_mode(family_count, pooled_min, narrative_min, allow_pooled)
        values = [float(study['effect_size']) for study in family_studies]
        heterogeneity_result = (
            heterogeneity_only_summary(random_effects_on_logit(family_studies))
            if family_count >= narrative_min
            else None
        )
        family_payload: dict[str, Any] = {
            'family': family,
            'study_count': family_count,
            'analysis_mode': mode,
            'analysis_action': {
                'pooled': 'pool_and_report',
                'narrative_forest_only': 'forest_and_narrative_only',
                'descriptive_only': 'descriptive_only',
            }[mode],
            'descriptive': {
                'mean': statistics.mean(values) if values else None,
                'median': statistics.median(values) if values else None,
                'min': min(values) if values else None,
                'max': max(values) if values else None,
            },
            'heterogeneity_result': heterogeneity_result,
            'pooled_result': None,
        }
        if mode == 'pooled':
            family_payload['pooled_result'] = heterogeneity_result

        family_results[family] = family_payload
        pooled_effect = (
            family_payload['pooled_result']['pooled_effect']
            if family_payload.get('pooled_result')
            else None
        )
        heterogeneity_i2 = (
            family_payload['heterogeneity_result']['heterogeneity']['I2']
            if family_payload.get('heterogeneity_result')
            else None
        )
        summary_rows.append(
            {
                'family': family,
                'study_count': family_count,
                'analysis_mode': mode,
                'descriptive_mean': round(family_payload['descriptive']['mean'], 6) if family_payload['descriptive']['mean'] is not None else '',
                'descriptive_median': round(family_payload['descriptive']['median'], 6) if family_payload['descriptive']['median'] is not None else '',
                'descriptive_min': round(family_payload['descriptive']['min'], 6) if family_payload['descriptive']['min'] is not None else '',
                'descriptive_max': round(family_payload['descriptive']['max'], 6) if family_payload['descriptive']['max'] is not None else '',
                'pooled_effect': round(float(pooled_effect), 6) if pooled_effect is not None else '',
                'heterogeneity_i2': round(float(heterogeneity_i2), 3) if heterogeneity_i2 is not None else '',
            }
        )
    return family_results, summary_rows


def select_primary_family(family_results: dict[str, Any]) -> dict[str, Any] | None:
    eligible = [
        item
        for item in family_results.values()
        if item.get('analysis_mode') in {'pooled', 'narrative_forest_only'} and item.get('heterogeneity_result')
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda item: (
            -int(item['study_count']),
            float(item['heterogeneity_result']['heterogeneity']['I2']),
            item['family'],
        )
    )
    selected = eligible[0]
    payload = {
        'family': selected['family'],
        'study_count': selected['study_count'],
        'analysis_mode': selected['analysis_mode'],
        'selection_rule': 'largest_n_then_lowest_I2',
    }
    if selected.get('pooled_result'):
        payload['pooled_result'] = selected['pooled_result']
    if selected.get('heterogeneity_result'):
        payload['heterogeneity_result'] = selected['heterogeneity_result']
    return payload


def subgroup_meta(studies: list[dict[str, Any]], subgroup_by: str) -> dict[str, dict[str, Any]]:
    if not subgroup_by:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for study in studies:
        subgroup = str(study.get('subgroup') or SUBGROUP_DEFAULT)
        grouped[subgroup].append(study)
    results: dict[str, dict[str, Any]] = {}
    for subgroup, subgroup_studies in sorted(grouped.items()):
        if len(subgroup_studies) < 2:
            continue
        results[subgroup] = random_effects_on_logit(subgroup_studies)
    return results


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_demo_rows() -> list[dict[str, Any]]:
    random.seed(42)
    families = ['accuracy', 'accuracy', 'accuracy', 'accuracy', 'accuracy', 'sensitivity', 'sensitivity', 'sensitivity', 'auc_auroc', 'f1']
    subsites = ['colorectal', 'gastric', 'esophageal', 'general_gi']
    rows: list[dict[str, Any]] = []
    for index in range(1, 31):
        family = random.choice(families)
        base = {
            'accuracy': 0.78,
            'sensitivity': 0.72,
            'specificity': 0.74,
            'auc_auroc': 0.81,
            'f1': 0.69,
        }[family]
        value = min(max(base + random.gauss(0, 0.06), 0.05), 0.98)
        sample_size = random.randint(80, 420)
        se = math.sqrt(value * (1.0 - value) / sample_size)
        rows.append(
            {
                'record_id': f'demo_{index:03d}',
                'study_label': f'Demo study {index}',
                'first_author': f'Author{index}',
                'metric_family': family,
                'metric': family,
                'effect_size': f'{value:.6f}',
                'se': f'{se:.6f}',
                'sample_size': str(sample_size),
                'gi_subsite': random.choice(subsites),
                'tier': random.choice(['I-a', 'I-b', 'II', 'III']),
                'crl': random.choice(['low', 'medium', 'high']),
                'wfs': random.choice(['triage', 'diagnosis', 'treatment planning']),
                'publication_year': str(random.choice([2022, 2023, 2024, 2025, 2026])),
            }
        )
    return rows


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.demo:
        return make_demo_rows()
    if not args.input:
        raise SystemExit('--input is required unless --demo is used')
    with open(args.input, 'r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def first_author_from_authors(authors: str) -> str:
    if not authors:
        return ''
    first = re.split(r';| and |\|', authors, maxsplit=1)[0].strip()
    first = re.sub(r'\([^)]*\)', '', first).strip()
    if not first:
        return ''
    if ',' in first:
        surname = first.split(',', 1)[0].strip()
        return surname
    tokens = [token.strip(' ,.') for token in first.split() if token.strip(' ,.')]
    if len(tokens) >= 2 and tokens[-2].lower() in {'da', 'de', 'del', 'den', 'der', 'di', 'du', 'la', 'le', 'van', 'von'}:
        return f'{tokens[-2]} {tokens[-1]}'
    return tokens[-1] if tokens else ''


def load_first_author_lookup(output_dir: pathlib.Path) -> dict[str, str]:
    table_path = output_dir.parent / 'figures' / 'table_1_characteristics_of_included_studies.csv'
    if not table_path.exists():
        return {}
    lookup: dict[str, str] = {}
    with open(table_path, 'r', encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            record_id = str(row.get('study_id') or row.get('record_id') or '').strip()
            first_author = str(row.get('first_author') or '').strip()
            if record_id and first_author and first_author.lower() != 'not reported':
                lookup[record_id] = first_author
    return lookup


def load_legacy_overall(output_dir: pathlib.Path, legacy_reference: str) -> dict[str, Any] | None:
    candidate_paths: list[pathlib.Path] = []
    if legacy_reference:
        candidate_paths.append(pathlib.Path(legacy_reference))
        candidate_paths.append(output_dir / 'meta_results.json')
    else:
        return None
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        for key in ('legacy_overall', 'overall'):
            overall = payload.get(key)
            if matches_locked_legacy_overall(overall):
                return build_locked_legacy_overall()
    return None


def main() -> None:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = load_rows(args)
    first_author_lookup = load_first_author_lookup(output_dir)
    for row in raw_rows:
        if str(row.get('first_author') or '').strip():
            continue
        record_id = str(row.get('record_id') or row.get('id') or row.get('study_id') or '').strip()
        inferred_first_author = first_author_lookup.get(record_id) or first_author_from_authors(str(row.get('authors') or '').strip())
        if inferred_first_author:
            row['first_author'] = inferred_first_author
    studies, skipped_records = extract_studies(raw_rows, args)
    if len(studies) < 2:
        raise SystemExit('Need at least two analyzable studies')

    family_results, family_summary_rows = summarize_family(
        studies=studies,
        pooled_min=args.family_pooled_min,
        narrative_min=args.family_narrative_min,
        allow_pooled=args.allow_pooled_family_effects,
    )
    primary_family = select_primary_family(family_results)

    overall_legacy = load_legacy_overall(output_dir, args.legacy_reference)
    subgroup_results = subgroup_meta(studies, args.subgroup_by)

    family_mode_counts = defaultdict(int)
    for item in family_results.values():
        family_mode_counts[item['analysis_mode']] += 1

    candidate_universe_count = sum(1 for study in studies if study.get('candidate_universe') in {'true', '1', 'yes'})
    meta_payload = {
        'analysis_identity': 'legacy_heterogeneous_synthesis_secondary',
        'effect_measure_input': args.effect_measure,
        'analysis_model': 'family-first proportion meta-analysis (logit random-effects)',
        'study_count': len(studies),
        'skipped_records': skipped_records,
        'family_rules': {
            'family_by_column': args.family_by,
            'pooled_threshold_n': args.family_pooled_min,
            'narrative_threshold_n': args.family_narrative_min,
            'pooling_policy': {
                'n>=pooled_threshold': 'pooled' if args.allow_pooled_family_effects else 'narrative_forest_only',
                'n_between_narrative_and_pooled': 'narrative_forest_only',
                'n< narrative_threshold': 'descriptive_only',
            },
        },
        'family_mode_counts': dict(family_mode_counts),
        'primary_family': primary_family,
        'candidate_universe': {
            'enabled': True,
            'study_count': candidate_universe_count,
        },
        'doi_plot': {
            'available': False,
            'lfk_index': None,
            'method': 'not computed in family-first pipeline',
        },
        'data_quality': {
            'sample_size_imputed_count': sum(1 for study in studies if study['sample_size_imputed']),
            'sample_size_reported_count': sum(1 for study in studies if not study['sample_size_imputed']),
        },
        'family_first_reference': {
            'json': 'family_results.json',
            'summary_csv': 'family_analysis_summary.csv',
            'study_csv': 'meta_study_table.csv',
            'pooled_csv': 'pooled_family_results.csv',
        },
    }
    if overall_legacy is not None:
        meta_payload['overall'] = {
            **overall_legacy,
            'status': 'legacy_heterogeneous_synthesis_secondary',
            'note': 'Historical heterogeneous pooled synthesis retained as a secondary legacy summary; not headline analysis.',
        }
        meta_payload['legacy_overall'] = {
            **overall_legacy,
            'status': 'legacy_heterogeneous_synthesis_secondary',
            'note': 'Historical heterogeneous pooled synthesis retained as a secondary legacy summary; not headline analysis.',
        }
    subgroup_payload = {
        'analysis_identity': 'legacy_subgroup_synthesis_secondary',
        'subgroup_by': args.subgroup_by or None,
        'results': subgroup_results,
        'note': 'Subgroup synthesis retained for compatibility and should be interpreted as secondary legacy output.',
    }
    family_payload = {
        'analysis_identity': 'family_first_exploratory_synthesis',
        'family_by': args.family_by,
        'thresholds': {
            'pooled_min': args.family_pooled_min,
            'narrative_min': args.family_narrative_min,
            'allow_pooled_family_effects': args.allow_pooled_family_effects,
        },
        'pooled_families': sorted([name for name, payload in family_results.items() if payload['analysis_mode'] == 'pooled']),
        'narrative_families': sorted([name for name, payload in family_results.items() if payload['analysis_mode'] == 'narrative_forest_only']),
        'descriptive_families': sorted([name for name, payload in family_results.items() if payload['analysis_mode'] == 'descriptive_only']),
        'results': family_results,
        'summary_rows': family_summary_rows,
    }

    (output_dir / 'meta_results.json').write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output_dir / 'subgroup_results.json').write_text(json.dumps(subgroup_payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output_dir / 'family_results.json').write_text(json.dumps(family_payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    write_csv(
        output_dir / 'family_analysis_summary.csv',
        [
            'family',
            'study_count',
            'analysis_mode',
            'descriptive_mean',
            'descriptive_median',
            'descriptive_min',
            'descriptive_max',
            'pooled_effect',
            'heterogeneity_i2',
        ],
        family_summary_rows,
    )
    study_table = []
    for study in studies:
        family_mode_value = family_results[study['metric_family']]['analysis_mode']
        study_table.append(
            {
                'record_id': study['record_id'],
                'study_label': study['study_label'],
                'first_author': study.get('first_author', 'Not reported'),
                'metric_family': study['metric_family'],
                'analysis_mode': family_mode_value,
                'effect_size': f"{study['effect_size']:.6f}",
                'se': f"{study['se']:.6f}",
                'sample_size': study['sample_size'],
                'sample_size_imputed': 'true' if study['sample_size_imputed'] else 'false',
                'candidate_universe': study.get('candidate_universe', 'true'),
                'gi_subsite': study.get('gi_subsite', ''),
                'tier': study.get('tier', ''),
                'crl': study.get('crl', ''),
                'wfs': study.get('wfs', ''),
                'publication_year': study.get('publication_year', ''),
            }
        )
    write_csv(
        output_dir / 'meta_study_table.csv',
        [
            'record_id',
            'study_label',
            'first_author',
            'metric_family',
            'analysis_mode',
            'effect_size',
            'se',
            'sample_size',
            'sample_size_imputed',
            'candidate_universe',
            'gi_subsite',
            'tier',
            'crl',
            'wfs',
            'publication_year',
        ],
        study_table,
    )

    pooled_rows = []
    for family, payload in sorted(family_results.items()):
        pooled = payload.get('pooled_result')
        if not pooled:
            continue
        pooled_rows.append(
            {
                'family': family,
                'study_count': payload['study_count'],
                'pooled_effect': f"{pooled['pooled_effect']:.6f}",
                'pooled_ci_lower': f"{pooled['pooled_ci_lower']:.6f}",
                'pooled_ci_upper': f"{pooled['pooled_ci_upper']:.6f}",
                'prediction_interval_lower': f"{pooled['heterogeneity']['prediction_interval_lower']:.6f}",
                'prediction_interval_upper': f"{pooled['heterogeneity']['prediction_interval_upper']:.6f}",
                'i2': f"{pooled['heterogeneity']['I2']:.3f}",
            }
        )
    write_csv(
        output_dir / 'pooled_family_results.csv',
        [
            'family',
            'study_count',
            'pooled_effect',
            'pooled_ci_lower',
            'pooled_ci_upper',
            'prediction_interval_lower',
            'prediction_interval_upper',
            'i2',
        ],
        pooled_rows,
    )

    print('Meta-analysis summary')
    print(f'- Analyzable studies: {len(studies)}')
    if overall_legacy is not None:
        print(f"- Legacy overall pooled effect: {meta_payload['overall']['pooled_effect']:.4f}")
        print(f"- Legacy overall 95% CI: [{meta_payload['overall']['pooled_ci_lower']:.4f}, {meta_payload['overall']['pooled_ci_upper']:.4f}]")
        print(f"- Legacy overall I^2: {meta_payload['overall']['heterogeneity']['I2']:.1f}%")
    if primary_family is not None:
        print(f"- Primary family: {primary_family['family']} (n={primary_family['study_count']})")
        if primary_family.get('pooled_result'):
            pooled = primary_family['pooled_result']
            print(f"- Primary pooled effect: {pooled['pooled_effect']:.4f} [{pooled['pooled_ci_lower']:.4f}, {pooled['pooled_ci_upper']:.4f}]")
        elif primary_family.get('heterogeneity_result'):
            heterogeneity = primary_family['heterogeneity_result']['heterogeneity']
            print(f"- Primary family heterogeneity: I^2={heterogeneity['I2']:.1f}%")
    print(f"- Family modes: {dict(family_mode_counts)}")
    print(f'- Outputs: {output_dir}')


if __name__ == '__main__':
    main()
