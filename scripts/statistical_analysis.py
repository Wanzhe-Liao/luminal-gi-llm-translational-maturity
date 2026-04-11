from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import random
from collections import Counter, defaultdict
from typing import Any

from build_evidence_map import (
    SimpleCanvas,
    draw_axis_frame,
    draw_label,
    draw_title_block,
    first_nonempty,
    generate_demo_rows,
    normalize_sample_size,
    parse_wfs,
    parse_year,
    save_canvas_outputs,
    write_csv,
)
from pipeline_lib import CRL_LEVELS, GI_SUBSITE_ORDER, TIERS, WFS_ORDER

OPEN_SOURCE_MODELS = ['llama', 'mistral', 'qwen', 'deepseek', 'falcon', 'bloom', 'vicuna', 'alpaca']
CLOSED_SOURCE_MODELS = ['gpt', 'chatgpt', 'claude', 'gemini', 'bard', 'palm', 'med-palm', 'copilot']
HIGH_EVIDENCE_TIERS = {'S', 'I-a', 'I-b'}
TIER_RANK = {'III': 0, 'II': 1, 'I-b': 2, 'I-a': 3, 'S': 4}
READINESS_STAGE_RANK = {
    'not_ready': 0,
    'external_validation_needed': 1,
    'human_review_only': 2,
    'prospective_trial_candidate': 3,
}
TRIPOD_ITEM_ORDER = [
    'title_identifies_llm',
    'abstract_structured',
    'objectives_stated',
    'data_source_described',
    'participants_described',
    'outcome_defined',
    'llm_selection_justified',
    'llm_version_reported',
    'prompt_engineering_described',
    'llm_config_reported',
    'sample_size_justified',
    'missing_data_addressed',
    'statistical_methods_described',
    'risk_of_bias_assessed',
    'participant_flow_reported',
    'performance_metrics_reported',
    'model_updating_discussed',
    'limitations_discussed',
    'clinical_implications_stated',
]
TRIPOD_EXTENDED_OPERATIONAL_ITEMS = [
    'llm_selection_justified',
    'prompt_engineering_described',
    'llm_config_reported',
    'risk_of_bias_assessed',
    'model_updating_discussed',
]
TRIPOD_CORE_OPERATIONAL_ITEMS = [item for item in TRIPOD_ITEM_ORDER if item not in TRIPOD_EXTENDED_OPERATIONAL_ITEMS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run descriptive and inferential analyses for the GI oncology LLM evidence map.')
    parser.add_argument('--input', help='Input labeled/extraction CSV')
    parser.add_argument('--tripod-input', help='Optional TRIPOD scoring CSV')
    parser.add_argument('--readiness-input', help='Optional readiness classification CSV')
    parser.add_argument('--output-dir', required=True, help='Directory for analysis outputs')
    parser.add_argument('--run-all', action='store_true', help='Run all analysis modules')
    parser.add_argument('--demo', action='store_true', help='Generate demo data and run full analysis')
    return parser.parse_args()


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 200
    epsilon = 3.0e-7
    min_float = 1.0e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c_term = 1.0
    d_term = 1.0 - qab * x / qap
    if abs(d_term) < min_float:
        d_term = min_float
    d_term = 1.0 / d_term
    h_term = d_term
    for index in range(1, max_iterations + 1):
        twice_index = 2 * index
        aa = index * (b - index) * x / ((qam + twice_index) * (a + twice_index))
        d_term = 1.0 + aa * d_term
        if abs(d_term) < min_float:
            d_term = min_float
        c_term = 1.0 + aa / c_term
        if abs(c_term) < min_float:
            c_term = min_float
        d_term = 1.0 / d_term
        h_term *= d_term * c_term
        aa = -(a + index) * (qab + index) * x / ((a + twice_index) * (qap + twice_index))
        d_term = 1.0 + aa * d_term
        if abs(d_term) < min_float:
            d_term = min_float
        c_term = 1.0 + aa / c_term
        if abs(c_term) < min_float:
            c_term = min_float
        d_term = 1.0 / d_term
        delta = d_term * c_term
        h_term *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h_term


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    beta_term = math.exp(a * math.log(x) + b * math.log1p(-x) - log_beta)
    threshold = (a + 1.0) / (a + b + 2.0)
    if x < threshold:
        return beta_term * beta_continued_fraction(a, b, x) / a
    return 1.0 - beta_term * beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_two_tailed_p_value(t_statistic: float, degrees_of_freedom: int) -> float:
    if degrees_of_freedom <= 0:
        return 1.0
    absolute_t = abs(t_statistic)
    if absolute_t == 0.0:
        return 1.0
    x_value = degrees_of_freedom / (degrees_of_freedom + absolute_t * absolute_t)
    p_value = regularized_incomplete_beta(x_value, degrees_of_freedom / 2.0, 0.5)
    return max(0.0, min(1.0, p_value))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_boolish(value: Any) -> bool | None:
    if value is True:
        return True
    if value is False:
        return False
    text = str(value or '').strip().lower()
    if not text or text == 'not_reported':
        return None
    if text in {'true', 'yes', '1', 'open'}:
        return True
    if text in {'false', 'no', '0', 'closed'}:
        return False
    return None


def classify_model_openness(model_name: str) -> str:
    lower = (model_name or '').strip().lower()
    if any(token in lower for token in OPEN_SOURCE_MODELS):
        return 'open_source'
    if any(token in lower for token in CLOSED_SOURCE_MODELS):
        return 'closed_source'
    return 'unknown'


def compute_transparency_score(record: dict[str, Any]) -> dict[str, Any]:
    model_version_exact = first_nonempty(record, 'model_version_exact')
    llm_model = first_nonempty(record, 'llm_model')
    prompt_reported = parse_boolish(record.get('prompt_reported')) is True
    temperature_reported = parse_boolish(record.get('temperature_reported')) is True
    rag_reported = parse_boolish(record.get('rag_reported')) is True
    data_availability = first_nonempty(record, 'data_availability', default='not_reported')
    code_availability = first_nonempty(record, 'code_availability', default='not_reported')
    external_validation = parse_boolish(record.get('external_validation')) is True

    model_points = 2 if model_version_exact and model_version_exact != 'not_reported' else (1 if llm_model and llm_model != 'not_reported' else 0)
    parameter_points = 1 if temperature_reported else 0
    prompt_points = 1 if prompt_reported else 0
    rag_points = 1 if rag_reported else 0
    data_points = {'open': 2, 'restricted': 1}.get(data_availability, 0)
    code_points = 1 if code_availability == 'open' else 0
    external_points = 2 if external_validation else 0
    total = model_points + parameter_points + prompt_points + rag_points + data_points + code_points + external_points
    return {
        'model_version_points': model_points,
        'parameter_points': parameter_points,
        'prompt_points': prompt_points,
        'rag_points': rag_points,
        'data_points': data_points,
        'code_points': code_points,
        'external_validation_points': external_points,
        'transparency_score': total,
    }


def normalize_records_extended(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in raw_rows:
        tier = first_nonempty(row, 'tier')
        crl = first_nonempty(row, 'crl')
        if not tier or not crl:
            continue
        year = parse_year(row)
        if year is None:
            continue
        wfs = [item for item in parse_wfs(first_nonempty(row, 'wfs')) if item in WFS_ORDER]
        sample_size_text = normalize_sample_size(first_nonempty(row, 'sample_size', default='not_reported'))
        sample_size = None
        if sample_size_text not in {'', 'not_reported'}:
            try:
                sample_size = int(float(sample_size_text))
            except ValueError:
                sample_size = None
        title = first_nonempty(row, 'title', default='Untitled study')
        abstract = first_nonempty(row, 'abstract')
        record = {
            'record_id': first_nonempty(row, 'record_id'),
            'protocol': first_nonempty(row, 'protocol', default='A'),
            'title': title,
            'abstract': abstract,
            'year': year,
            'tier': tier,
            'crl': crl,
            'wfs': wfs,
            'gi_subsite': first_nonempty(row, 'gi_subsite', default='general_gi'),
            'llm_model': first_nonempty(row, 'llm_model', default='not_reported'),
            'model_openness': classify_model_openness(first_nonempty(row, 'llm_model', default='')),
            'sample_size': sample_size,
            'peer_review_status': first_nonempty(row, 'peer_review_status', default='peer_reviewed'),
            'region': first_nonempty(row, 'region', 'country_region', 'country', default='not_reported'),
            'data_availability': first_nonempty(row, 'data_availability', default='not_reported'),
            'code_availability': first_nonempty(row, 'code_availability', default='not_reported'),
            'external_validation': row.get('external_validation', 'not_reported'),
            'prompt_reported': row.get('prompt_reported', False),
            'temperature_reported': row.get('temperature_reported', False),
            'rag_reported': row.get('rag_reported', False),
            'model_version_exact': first_nonempty(row, 'model_version_exact', default='not_reported'),
        }
        record.update(compute_transparency_score(record))
        records.append(record)
    return records


def build_demo_raw_rows(total: int = 60) -> list[dict[str, Any]]:
    random.seed(42)
    rows = generate_demo_rows(total)
    regions = ['Asia', 'Europe', 'North America', 'Oceania']
    designs = ['retrospective cohort', 'prospective', 'cross-sectional', 'RCT', 'simulated', 'exam_based']
    for row in rows:
        row['protocol'] = random.choices(['A', 'B'], weights=[4, 1], k=1)[0]
        row['region'] = random.choice(regions)
        row['study_design'] = random.choice(designs)
        row['peer_review_status'] = 'peer_reviewed' if row['protocol'] == 'A' else random.choice(['peer_reviewed', 'mixed_or_non_peer_reviewed'])
        row['model_version_exact'] = 'GPT-4-turbo-2024-04-09' if 'GPT' in row.get('llm_model', '') and random.random() < 0.5 else 'not_reported'
        row['prompt_reported'] = random.choice([True, False, False])
        row['temperature_reported'] = random.choice([True, False, False])
        row['rag_reported'] = random.choice([True, False, False, False])
        row['data_availability'] = random.choice(['open', 'restricted', 'closed', 'not_reported'])
        row['code_availability'] = random.choice(['open', 'closed', 'not_reported'])
        row['external_validation'] = random.choice([True, False, 'not_reported'])
    return rows


def fit_log_linear_trend(year_counts: dict[int, int]) -> dict[str, Any]:
    years = sorted(year_counts)
    if len(years) < 2:
        return {'status': 'insufficient_years'}
    x = [year - min(years) for year in years]
    y = [math.log(year_counts[year] + 0.5) for year in years]
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    sxx = sum((value - x_mean) ** 2 for value in x)
    if sxx <= 0:
        return {'status': 'constant_year_index'}
    sxy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    beta = sxy / sxx
    alpha = y_mean - beta * x_mean
    fitted = [alpha + beta * xi for xi in x]
    residuals = [yi - fi for yi, fi in zip(y, fitted)]
    mse = (sum(value * value for value in residuals) / max(1, n - 2)) if n > 2 else 0.0
    se_beta = math.sqrt(mse / sxx) if n > 2 and sxx > 0 else 0.0
    z_score = (beta / se_beta) if se_beta > 0 else 0.0
    p_value = 2.0 * (1.0 - normal_cdf(abs(z_score))) if se_beta > 0 else 1.0
    irr = math.exp(beta)
    ci_low = math.exp(beta - 1.96 * se_beta) if se_beta > 0 else irr
    ci_high = math.exp(beta + 1.96 * se_beta) if se_beta > 0 else irr
    return {
        'status': 'ok',
        'years': years,
        'counts': [year_counts[year] for year in years],
        'irr': round(irr, 4),
        'ci_95_low': round(ci_low, 4),
        'ci_95_high': round(ci_high, 4),
        'p_value': round(p_value, 6),
    }


def load_optional_csv(path: str | None) -> list[dict[str, str]]:
    if not path:
        return []
    csv_path = pathlib.Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def rankdata(values: list[float]) -> list[float]:
    if not values:
        return []
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and values[ordered[end + 1]] == values[ordered[index]]:
            end += 1
        average_rank = (index + end + 2) / 2.0
        for rank_index in range(index, end + 1):
            ranks[ordered[rank_index]] = average_rank
        index = end + 1
    return ranks


def pearson_correlation(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return float('nan')
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b))
    denominator = math.sqrt(sum((a - mean_a) ** 2 for a in values_a) * sum((b - mean_b) ** 2 for b in values_b))
    if denominator == 0:
        return float('nan')
    return numerator / denominator


def spearman_summary(values_a: list[float], values_b: list[float]) -> dict[str, Any]:
    if len(values_a) != len(values_b) or len(values_a) < 3:
        return {'status': 'insufficient_data'}
    rho = pearson_correlation(rankdata(values_a), rankdata(values_b))
    if math.isnan(rho):
        return {'status': 'undefined'}
    n = len(values_a)
    t_statistic = 0.0
    if abs(rho) >= 0.999999:
        p_value = 0.0
        t_statistic = math.copysign(float('inf'), rho)
    else:
        denominator = max(1e-12, 1.0 - rho * rho)
        t_statistic = rho * math.sqrt((n - 2.0) / denominator)
        p_value = student_t_two_tailed_p_value(t_statistic, n - 2)
    return {
        'status': 'ok',
        'n': n,
        'rho': round(rho, 4),
        'p_value_normal_approx': round(max(0.0, min(1.0, p_value)), 6),
        'p_value_t_approx': round(max(0.0, min(1.0, p_value)), 6),
        't_statistic': round(t_statistic, 4) if math.isfinite(t_statistic) else None,
        'df': n - 2,
        'method_note': 'Spearman rho with t-distribution approximation; exploratory and non-causal.',
    }


def compute_tripod_dual_track_summary(tripod_rows: list[dict[str, str]]) -> dict[str, Any]:
    scored_rows = [row for row in tripod_rows if first_nonempty(row, 'total_score') not in {'', 'not_reported'}]
    if not scored_rows:
        return {'status': 'unavailable'}
    full_scores = [float(first_nonempty(row, 'total_score')) for row in scored_rows]
    core_scores = [sum(int(first_nonempty(row, item, default='0')) for item in TRIPOD_CORE_OPERATIONAL_ITEMS) for row in scored_rows]
    item_rates = []
    denominator = len(scored_rows)
    for item in TRIPOD_ITEM_ORDER:
        reported = sum(1 for row in scored_rows if first_nonempty(row, item, default='0') in {'1', 'true', 'yes'})
        item_rates.append(
            {
                'tripod_item': item,
                'item_group': 'core' if item in TRIPOD_CORE_OPERATIONAL_ITEMS else 'extended',
                'reporting_rate_percent': round(reported * 100 / denominator, 2),
            }
        )
    core_item_rates = [row['reporting_rate_percent'] for row in item_rates if row['item_group'] == 'core']
    extended_item_rates = [row['reporting_rate_percent'] for row in item_rates if row['item_group'] == 'extended']
    return {
        'status': 'ok',
        'total_included_records': len(tripod_rows),
        'scored_records': denominator,
        'operational_definition': {
            'full_item_count': len(TRIPOD_ITEM_ORDER),
            'core_item_count': len(TRIPOD_CORE_OPERATIONAL_ITEMS),
            'extended_item_count': len(TRIPOD_EXTENDED_OPERATIONAL_ITEMS),
            'core_items': TRIPOD_CORE_OPERATIONAL_ITEMS,
            'extended_items': TRIPOD_EXTENDED_OPERATIONAL_ITEMS,
            'note': 'Core-14 is an operational subset of items considered broadly applicable across study/task categories in this abstract-level surveillance implementation.',
        },
        'full_19_item': {
            'mean_total_score': round(sum(full_scores) / len(full_scores), 4),
            'median_total_score': round(quantile(full_scores, 0.5), 3),
            'mean_percent': round((sum(full_scores) / len(full_scores)) * 100.0 / len(TRIPOD_ITEM_ORDER), 2),
        },
        'core_14_item_operational': {
            'mean_total_score': round(sum(core_scores) / len(core_scores), 4),
            'median_total_score': round(quantile(core_scores, 0.5), 3),
            'mean_percent': round((sum(core_scores) / len(core_scores)) * 100.0 / len(TRIPOD_CORE_OPERATIONAL_ITEMS), 2),
        },
        'item_group_summary': {
            'core_mean_reporting_rate_percent': round(sum(core_item_rates) / len(core_item_rates), 2) if core_item_rates else None,
            'extended_mean_reporting_rate_percent': round(sum(extended_item_rates) / len(extended_item_rates), 2) if extended_item_rates else None,
        },
        'item_reporting_rates': item_rates,
    }


def compute_maturity_trend_analysis(
    study_records: list[dict[str, Any]],
    tripod_rows: list[dict[str, str]],
    readiness_rows: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    study_lookup = {str(record['record_id']): record for record in study_records}
    study_years = [float(record['year']) for record in study_records]
    study_tier_ranks = [float(TIER_RANK.get(record['tier'], 0)) for record in study_records]

    tripod_scored = [row for row in tripod_rows if first_nonempty(row, 'total_score') not in {'', 'not_reported'}]
    tripod_points = []
    for row in tripod_scored:
        record_id = first_nonempty(row, 'record_id')
        if record_id not in study_lookup:
            continue
        core_score = sum(int(first_nonempty(row, item, default='0')) for item in TRIPOD_CORE_OPERATIONAL_ITEMS)
        tripod_points.append({'record_id': record_id, 'year': float(study_lookup[record_id]['year']), 'core_score': float(core_score)})

    readiness_points = []
    for row in readiness_rows:
        record_id = first_nonempty(row, 'record_id')
        if record_id not in study_lookup:
            continue
        stage = first_nonempty(row, 'readiness_stage', 'readiness_class', default='not_ready')
        readiness_points.append({'record_id': record_id, 'year': float(study_lookup[record_id]['year']), 'rank': float(READINESS_STAGE_RANK.get(stage, 0))})

    yearly: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            'year': 0,
            'study_count': 0,
            'tier_rank_values': [],
            'high_evidence_count': 0,
            'tripod_core_scores': [],
            'use_case_count': 0,
            'readiness_rank_values': [],
            'readiness_stage_counts': Counter(),
        }
    )
    for record in study_records:
        year = int(record['year'])
        block = yearly[year]
        block['year'] = year
        block['study_count'] += 1
        block['tier_rank_values'].append(float(TIER_RANK.get(record['tier'], 0)))
        if record['tier'] in HIGH_EVIDENCE_TIERS:
            block['high_evidence_count'] += 1
    for point in tripod_points:
        year = int(point['year'])
        yearly[year]['tripod_core_scores'].append(point['core_score'])
    for point in readiness_points:
        year = int(point['year'])
        yearly[year]['use_case_count'] += 1
        yearly[year]['readiness_rank_values'].append(point['rank'])
    for row in readiness_rows:
        record_id = first_nonempty(row, 'record_id')
        if record_id not in study_lookup:
            continue
        year = int(study_lookup[record_id]['year'])
        stage = first_nonempty(row, 'readiness_stage', 'readiness_class', default='not_ready')
        yearly[year]['readiness_stage_counts'][stage] += 1

    yearly_rows: list[dict[str, Any]] = []
    for year in sorted(yearly):
        block = yearly[year]
        study_count = block['study_count'] or 1
        use_case_count = block['use_case_count'] or 1
        yearly_rows.append(
            {
                'year': year,
                'study_count': block['study_count'],
                'tier_mean_rank': round(sum(block['tier_rank_values']) / len(block['tier_rank_values']), 4) if block['tier_rank_values'] else None,
                'high_evidence_share_percent': round(block['high_evidence_count'] * 100.0 / study_count, 2),
                'tripod_core_mean_score': round(sum(block['tripod_core_scores']) / len(block['tripod_core_scores']), 4) if block['tripod_core_scores'] else None,
                'tripod_core_mean_percent': round((sum(block['tripod_core_scores']) / len(block['tripod_core_scores'])) * 100.0 / len(TRIPOD_CORE_OPERATIONAL_ITEMS), 2) if block['tripod_core_scores'] else None,
                'use_case_count': block['use_case_count'],
                'readiness_mean_rank': round(sum(block['readiness_rank_values']) / len(block['readiness_rank_values']), 4) if block['readiness_rank_values'] else None,
                'readiness_higher_stage_share_percent': round(sum(1 for value in block['readiness_rank_values'] if value >= 2) * 100.0 / use_case_count, 2) if block['readiness_rank_values'] else None,
                'not_ready_use_case_count': block['readiness_stage_counts'].get('not_ready', 0),
                'external_validation_use_case_count': block['readiness_stage_counts'].get('external_validation_needed', 0),
                'human_review_use_case_count': block['readiness_stage_counts'].get('human_review_only', 0),
                'prospective_trial_use_case_count': block['readiness_stage_counts'].get('prospective_trial_candidate', 0),
            }
        )

    trend_payload = {
        'analysis_identity': 'maturity_over_time_exploratory_noncausal',
        'study_level': {
            'unit_of_analysis': 'study',
            'year_to_tier_rank': spearman_summary(study_years, study_tier_ranks),
            'yearly_descriptives': yearly_rows,
        },
        'tripod_core': {
            'unit_of_analysis': 'study_with_scored_tripod_record',
            'year_to_core14_score': spearman_summary([point['year'] for point in tripod_points], [point['core_score'] for point in tripod_points]),
        },
        'readiness_use_case': {
            'unit_of_analysis': 'use_case',
            'year_to_readiness_rank': spearman_summary([point['year'] for point in readiness_points], [point['rank'] for point in readiness_points]),
        },
        'methods_note': 'Exploratory, non-causal time-trend analysis using study-level tier rank, study-level TRIPOD core-14 score, and use-case-level readiness rank.',
    }
    return trend_payload, yearly_rows


def chi_square_p_value(statistic: float, df: int) -> float:
    if df <= 0:
        return 1.0
    z = ((statistic / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return max(0.0, min(1.0, 1.0 - normal_cdf(z)))


def fisher_exact_2x2(table: list[list[int]]) -> float:
    a, b = table[0]
    c, d = table[1]
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2

    def log_comb(n: int, k: int) -> float:
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    def probability(x: int) -> float:
        return math.exp(log_comb(col1, x) + log_comb(total - col1, row1 - x) - log_comb(total, row1))

    low = max(0, row1 - (total - col1))
    high = min(row1, col1)
    observed = probability(a)
    p_value = 0.0
    for value in range(low, high + 1):
        prob = probability(value)
        if prob <= observed + 1e-12:
            p_value += prob
    return min(1.0, p_value)


def contingency_test(table: list[list[int]]) -> dict[str, Any]:
    rows = len(table)
    cols = len(table[0]) if rows else 0
    if rows == 2 and cols == 2:
        return {'method': 'fisher_exact', 'p_value': round(fisher_exact_2x2(table), 6)}
    row_totals = [sum(row) for row in table]
    col_totals = [sum(table[row_index][col_index] for row_index in range(rows)) for col_index in range(cols)]
    total = sum(row_totals)
    statistic = 0.0
    for row_index in range(rows):
        for col_index in range(cols):
            expected = (row_totals[row_index] * col_totals[col_index] / total) if total else 0.0
            if expected > 0:
                statistic += (table[row_index][col_index] - expected) ** 2 / expected
    df = max(1, (rows - 1) * (cols - 1))
    return {'method': 'chi_square', 'statistic': round(statistic, 4), 'df': df, 'p_value': round(chi_square_p_value(statistic, df), 6)}


def mann_whitney_u(values_a: list[float], values_b: list[float]) -> dict[str, Any]:
    if not values_a or not values_b:
        return {'status': 'insufficient_data'}
    combined = sorted([(value, 'A') for value in values_a] + [(value, 'B') for value in values_b], key=lambda item: item[0])
    ranks = []
    index = 0
    while index < len(combined):
        end = index
        while end < len(combined) and combined[end][0] == combined[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        for tie_index in range(index, end):
            ranks.append((combined[tie_index][0], combined[tie_index][1], avg_rank))
        index = end
    rank_sum_a = sum(rank for _, group, rank in ranks if group == 'A')
    n1 = len(values_a)
    n2 = len(values_b)
    u1 = rank_sum_a - n1 * (n1 + 1) / 2.0
    mean_u = n1 * n2 / 2.0
    sd_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (u1 - mean_u) / sd_u if sd_u > 0 else 0.0
    p_value = 2.0 * (1.0 - normal_cdf(abs(z))) if sd_u > 0 else 1.0
    return {'status': 'ok', 'u_statistic': round(u1, 4), 'p_value': round(p_value, 6)}


def time_trend_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    overall_counts = Counter(record['year'] for record in records)
    by_tier = {}
    by_subsite = {}
    for tier in TIERS:
        counts = Counter(record['year'] for record in records if record['tier'] == tier)
        if counts:
            by_tier[tier] = fit_log_linear_trend(dict(counts))
    for subsite in GI_SUBSITE_ORDER:
        counts = Counter(record['year'] for record in records if record['gi_subsite'] == subsite)
        if counts:
            by_subsite[subsite] = fit_log_linear_trend(dict(counts))
    return {'overall': fit_log_linear_trend(dict(overall_counts)), 'by_tier': by_tier, 'by_subsite': by_subsite}


def protocol_comparison_authority_mirror(output_dir: pathlib.Path) -> dict[str, Any]:
    submission_summary_path = output_dir.parent / 'protocol_comparison' / 'submission_facing_summary.json'
    internal_authority_path = output_dir.parent / 'protocol_comparison' / 'comparison_summary.json'
    if submission_summary_path.exists():
        authority = json.loads(submission_summary_path.read_text(encoding='utf-8'))
        return {
            'status': 'submission_facing_mirror',
            'submission_facing_truth_source': 'protocol_comparison/submission_facing_summary.json',
            'authority_source': str(submission_summary_path),
            'internal_authority_source': str(internal_authority_path),
            'expanded_peer_reviewed_comparison_set_count': authority.get('expanded_peer_reviewed_comparison_set_count'),
            'supplementary_publication_status_records_reviewed': authority.get('supplementary_publication_status_records_reviewed'),
            'supplementary_publication_status_records_unique': authority.get('supplementary_publication_status_records_unique'),
            'supplementary_publication_status_records_overlap': authority.get('supplementary_publication_status_records_overlap'),
            'supplementary_publication_status_signature': authority.get('supplementary_publication_status_signature'),
            'descriptive_comparison_only': authority.get('descriptive_comparison_only', True),
        }
    if not internal_authority_path.exists():
        return {
            'status': 'missing_protocol_comparison_submission_summary',
            'submission_facing_truth_source': 'protocol_comparison/submission_facing_summary.json',
            'authority_source': str(submission_summary_path),
            'internal_authority_source': str(internal_authority_path),
        }
    authority = json.loads(internal_authority_path.read_text(encoding='utf-8'))
    overlap = authority.get('overlap') or {}
    return {
        'status': 'legacy_raw_authority_fallback',
        'submission_facing_truth_source': 'protocol_comparison/submission_facing_summary.json',
        'authority_source': str(submission_summary_path),
        'internal_authority_source': str(internal_authority_path),
        'expanded_peer_reviewed_comparison_set_count': authority.get('protocol_a_total'),
        'supplementary_publication_status_records_reviewed': authority.get('protocol_b_total'),
        'supplementary_publication_status_records_unique': authority.get('protocol_b_unique_records'),
        'supplementary_publication_status_records_overlap': overlap.get('overlap'),
        'supplementary_publication_status_signature': (
            f"{authority.get('protocol_b_total', 0)} reviewed / "
            f"{authority.get('protocol_b_unique_records', 0)} unique / "
            f"{overlap.get('overlap', 0)} overlap"
        ),
        'descriptive_comparison_only': True,
    }


def subgroup_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    subgroup = {
        'gi_subsite': {},
        'region': {},
        'model_openness': {},
        'year': {},
    }
    for subsite in GI_SUBSITE_ORDER:
        subset = [record for record in records if record['gi_subsite'] == subsite]
        subgroup['gi_subsite'][subsite] = {'count': len(subset), 'high_evidence_count': sum(1 for record in subset if record['tier'] in HIGH_EVIDENCE_TIERS)}
    for region in sorted({record['region'] for record in records}):
        subset = [record for record in records if record['region'] == region]
        subgroup['region'][region] = {'count': len(subset), 'high_evidence_count': sum(1 for record in subset if record['tier'] in HIGH_EVIDENCE_TIERS)}
    for openness in ['open_source', 'closed_source', 'unknown']:
        subset = [record for record in records if record['model_openness'] == openness]
        subgroup['model_openness'][openness] = {'count': len(subset), 'high_evidence_count': sum(1 for record in subset if record['tier'] in HIGH_EVIDENCE_TIERS)}
    for year in sorted({record['year'] for record in records}):
        subset = [record for record in records if record['year'] == year]
        subgroup['year'][str(year)] = {'count': len(subset), 'high_evidence_count': sum(1 for record in subset if record['tier'] in HIGH_EVIDENCE_TIERS)}
    return subgroup


def sensitivity_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    high_evidence = [record for record in records if record['tier'] in HIGH_EVIDENCE_TIERS]
    no_tier_iii = [record for record in records if record['tier'] != 'III']
    peer_reviewed_only = [record for record in records if record['peer_review_status'] == 'peer_reviewed']
    gist_net = [record for record in records if any(token in f"{record['title']} {record['abstract']}".lower() for token in ['gist', 'gi-net', 'neuroendocrine'])]
    return {
        'high_evidence_subset': {'count': len(high_evidence), 'wfs_counts': Counter(wfs for record in high_evidence for wfs in record['wfs'])},
        'exclude_tier_iii': {'count': len(no_tier_iii), 'wfs_counts': Counter(wfs for record in no_tier_iii for wfs in record['wfs'])},
        'peer_reviewed_only': {'count': len(peer_reviewed_only), 'wfs_counts': Counter(wfs for record in peer_reviewed_only for wfs in record['wfs'])},
        'gist_gi_net_lower_bound': {'count': len(gist_net), 'note': 'Computed within available dataset only; does not add external records.'},
    }


def sample_size_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for tier in TIERS:
        for subsite in GI_SUBSITE_ORDER:
            values = [record['sample_size'] for record in records if record['tier'] == tier and record['gi_subsite'] == subsite and isinstance(record['sample_size'], int)]
            if not values:
                continue
            ordered = sorted(values)
            median = quantile([float(value) for value in ordered], 0.5)
            p25 = quantile([float(value) for value in ordered], 0.25)
            p75 = quantile([float(value) for value in ordered], 0.75)
            rows.append({
                'tier': tier,
                'gi_subsite': subsite,
                'n': len(values),
                'median': round(median, 2),
                'iqr_low': round(p25, 2),
                'iqr_high': round(p75, 2),
                'lt_100_ratio': round(sum(1 for value in values if value < 100) / len(values), 4),
            })
    return rows


def render_time_trend_chart(base_path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    canvas = SimpleCanvas(1500, 920)
    draw_title_block(canvas, 'TIME TREND', 'PUBLICATION COUNTS BY YEAR')
    left, top, width, height = 120, 150, 1230, 620
    draw_axis_frame(canvas, left, top, width, height)
    years = sorted({record['year'] for record in records})
    counts = [sum(1 for record in records if record['year'] == year) for year in years]
    max_count = max(counts) if counts else 1
    if not years:
        save_canvas_outputs(canvas, base_path)
        return
    for idx, year in enumerate(years):
        x = left + int(width * idx / max(1, len(years) - 1)) if len(years) > 1 else left + width // 2
        y = top + height - int(height * counts[idx] / max_count)
        canvas.fill_rect(x - 8, y - 8, 16, 16, (53, 92, 125))
        if idx > 0:
            prev_x = left + int(width * (idx - 1) / max(1, len(years) - 1)) if len(years) > 1 else x
            prev_y = top + height - int(height * counts[idx - 1] / max_count)
            canvas.draw_line(prev_x, prev_y, x, y, (53, 92, 125))
        draw_label(canvas, x - 12, top + height + 24, str(year), scale=2)
        draw_label(canvas, x - 6, y - 28, str(counts[idx]), scale=2)
    save_canvas_outputs(canvas, base_path)


def render_transparency_distribution(base_path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    canvas = SimpleCanvas(1500, 920)
    draw_title_block(canvas, 'TRANSPARENCY SCORE', 'DISTRIBUTION OF 0 TO 10 INDEX')
    left, top, width, height = 120, 150, 1230, 620
    draw_axis_frame(canvas, left, top, width, height)
    counts = Counter(record['transparency_score'] for record in records)
    max_count = max(counts.values()) if counts else 1
    bar_group = width / 11
    for score in range(11):
        count = counts.get(score, 0)
        bar_h = int(height * count / max_count)
        x = left + int(bar_group * score + bar_group * 0.15)
        canvas.fill_rect(x, top + height - bar_h, int(bar_group * 0.65), bar_h, (88, 129, 87))
        draw_label(canvas, x + 10, top + height + 24, str(score), scale=2)
        draw_label(canvas, x + 4, top + height - bar_h - 24, str(count), scale=2)
    save_canvas_outputs(canvas, base_path)


def render_sample_size_distribution(base_path: pathlib.Path, records: list[dict[str, Any]]) -> None:
    canvas = SimpleCanvas(1600, 980)
    draw_title_block(canvas, 'SAMPLE SIZE', 'MEDIAN AND IQR BY EVIDENCE TIER')
    left, top, width, height = 140, 160, 1300, 650
    draw_axis_frame(canvas, left, top, width, height)
    tier_values = {tier: sorted(record['sample_size'] for record in records if record['tier'] == tier and isinstance(record['sample_size'], int)) for tier in TIERS}
    max_value = max((max(values) for values in tier_values.values() if values), default=1)
    group = width / len(TIERS)
    for idx, tier in enumerate(TIERS):
        values = tier_values[tier]
        x = left + int(group * idx + group * 0.4)
        draw_label(canvas, x - 20, top + height + 24, tier.replace('-', ''), scale=2)
        if not values:
            continue
        q1 = quantile([float(v) for v in values], 0.25)
        med = quantile([float(v) for v in values], 0.5)
        q3 = quantile([float(v) for v in values], 0.75)
        low = min(values)
        high = max(values)
        def y_pos(v: float) -> int:
            return top + height - int(height * v / max_value)
        canvas.draw_line(x, y_pos(low), x, y_pos(high), (70, 70, 70))
        canvas.fill_rect(x - 26, y_pos(q3), 52, max(2, y_pos(q1) - y_pos(q3)), (210, 172, 76))
        canvas.draw_line(x - 26, y_pos(med), x + 26, y_pos(med), (40, 40, 40))
    save_canvas_outputs(canvas, base_path)


def main() -> None:
    args = parse_args()
    if not args.demo and not args.input:
        raise SystemExit('--input is required unless --demo is used')
    if not args.demo and not args.run_all:
        raise SystemExit('--run-all is required for non-demo execution')

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        raw_rows = build_demo_raw_rows(60)
        tripod_rows: list[dict[str, str]] = []
        readiness_rows: list[dict[str, str]] = []
    else:
        import csv
        with open(args.input, 'r', encoding='utf-8-sig', newline='') as handle:
            raw_rows = list(csv.DictReader(handle))
        tripod_rows = load_optional_csv(args.tripod_input)
        readiness_rows = load_optional_csv(args.readiness_input)

    records = normalize_records_extended(raw_rows)
    if not records:
        raise SystemExit('No analyzable records found')

    trend = time_trend_analysis(records)
    protocol = protocol_comparison_authority_mirror(output_dir)
    subgroup = subgroup_analysis(records)
    sensitivity = sensitivity_analysis(records)
    sample_rows = sample_size_summary(records)
    tripod_dual_track = compute_tripod_dual_track_summary(tripod_rows)
    maturity_trends, maturity_yearly_rows = compute_maturity_trend_analysis(records, tripod_rows, readiness_rows)

    transparency_rows = []
    for record in records:
        transparency_rows.append({
            'record_id': record['record_id'],
            'protocol': record['protocol'],
            'year': record['year'],
            'tier': record['tier'],
            'gi_subsite': record['gi_subsite'],
            'llm_model': record['llm_model'],
            'transparency_score': record['transparency_score'],
            'model_version_points': record['model_version_points'],
            'parameter_points': record['parameter_points'],
            'prompt_points': record['prompt_points'],
            'rag_points': record['rag_points'],
            'data_points': record['data_points'],
            'code_points': record['code_points'],
            'external_validation_points': record['external_validation_points'],
        })

    yearly_rows = []
    for year in sorted({record['year'] for record in records}):
        yearly_rows.append({'year': year, 'study_count': sum(1 for record in records if record['year'] == year)})

    write_csv(output_dir / 'transparency_scores.csv', list(transparency_rows[0].keys()), transparency_rows)
    write_csv(output_dir / 'sample_size_summary.csv', ['tier', 'gi_subsite', 'n', 'median', 'iqr_low', 'iqr_high', 'lt_100_ratio'], sample_rows or [])
    write_csv(output_dir / 'yearly_counts.csv', ['year', 'study_count'], yearly_rows)
    write_csv(
        output_dir / 'maturity_yearly_summary.csv',
        [
            'year',
            'study_count',
            'tier_mean_rank',
            'high_evidence_share_percent',
            'tripod_core_mean_score',
            'tripod_core_mean_percent',
            'use_case_count',
            'readiness_mean_rank',
            'readiness_higher_stage_share_percent',
            'not_ready_use_case_count',
            'external_validation_use_case_count',
            'human_review_use_case_count',
            'prospective_trial_use_case_count',
        ],
        maturity_yearly_rows,
    )
    write_csv(
        output_dir / 'tripod_item_grouping.csv',
        ['tripod_item', 'item_group', 'reporting_rate_percent'],
        tripod_dual_track.get('item_reporting_rates', []),
    )

    (output_dir / 'time_trend_regression.json').write_text(json.dumps(trend, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output_dir / 'protocol_comparison.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output_dir / 'subgroup_analysis.json').write_text(json.dumps(subgroup, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output_dir / 'sensitivity_analysis.json').write_text(json.dumps(sensitivity, indent=2, ensure_ascii=False, default=lambda o: dict(o)) + '\n', encoding='utf-8')
    (output_dir / 'tripod_dual_track_summary.json').write_text(json.dumps(tripod_dual_track, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output_dir / 'maturity_trends.json').write_text(json.dumps(maturity_trends, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    render_time_trend_chart(output_dir / 'time_trend', records)
    render_transparency_distribution(output_dir / 'transparency_distribution', records)
    render_sample_size_distribution(output_dir / 'sample_size_distribution', records)

    report = {
        'record_count': len(records),
        'authority_mirrors': {
            'protocol_comparison': {
                'status': protocol.get('status'),
                'mirror_json': str(output_dir / 'protocol_comparison.json'),
                'authority_source': protocol.get('authority_source'),
            },
        },
        'outputs': {
            'time_trend_regression': str(output_dir / 'time_trend_regression.json'),
            'transparency_scores': str(output_dir / 'transparency_scores.csv'),
            'protocol_comparison': str(output_dir / 'protocol_comparison.json'),
            'subgroup_analysis': str(output_dir / 'subgroup_analysis.json'),
            'sensitivity_analysis': str(output_dir / 'sensitivity_analysis.json'),
            'sample_size_summary': str(output_dir / 'sample_size_summary.csv'),
            'tripod_dual_track_summary': str(output_dir / 'tripod_dual_track_summary.json'),
            'tripod_item_grouping': str(output_dir / 'tripod_item_grouping.csv'),
            'maturity_trends': str(output_dir / 'maturity_trends.json'),
            'maturity_yearly_summary': str(output_dir / 'maturity_yearly_summary.csv'),
            'time_trend_chart': str(output_dir / 'time_trend.png'),
            'transparency_chart': str(output_dir / 'transparency_distribution.png'),
            'sample_size_chart': str(output_dir / 'sample_size_distribution.png'),
        },
    }
    (output_dir / 'statistical_analysis_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(f'[statistical_analysis] report={output_dir / "statistical_analysis_report.json"}')
    print(f'[statistical_analysis] transparency={output_dir / "transparency_scores.csv"}')
    print(f'[statistical_analysis] sample_size={output_dir / "sample_size_summary.csv"}')
    print(f'[statistical_analysis] maturity={output_dir / "maturity_trends.json"}')


if __name__ == '__main__':
    main()
