"""Prepare family-aware meta-analysis inputs from extraction-level records.

This script standardizes performance metrics into homogeneous metric families
and emits a structured candidate universe for downstream meta-analysis.

Primary goals:
1) family-first grouping (accuracy/sensitivity/specificity/auc_auroc/f1),
2) bounded proportion values in (0, 1),
3) explicit sample-size imputation flags,
4) enrichment of tier/CRL/WFS fields from tier-labeled outputs.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import pathlib
import re
from collections import Counter
from typing import Any


PROPORTION_EPS = 1e-6

METRIC_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ('auc_auroc', ('auroc', 'auc', 'area under curve', 'area under the curve')),
    ('sensitivity', ('sensitivity', 'recall', 'true positive rate', 'tpr')),
    ('specificity', ('specificity', 'true negative rate', 'tnr')),
    ('f1', ('f1', 'f-1', 'f score', 'f-score')),
    ('accuracy', ('accuracy', 'acc', 'percent agreement', 'agreement rate', 'balanced accuracy')),
]

GI_SUBSITE_KEYWORDS = {
    'colorectal': ['colorectal', 'colon', 'rectal', 'rectum', 'crc'],
    'gastric': ['gastric', 'stomach'],
    'esophageal': ['esophag', 'oesophag'],
    'small_bowel': ['small bowel', 'small intestine', 'duodenum', 'jejunum', 'ileum'],
    'anal': ['anal', 'anus'],
    'multiple_gi': ['multiple'],
    'general_gi': ['gastrointestinal', 'digestive', 'gi cancer', 'gi oncol'],
}

TIER_FIELDS = ('tier', 'crl', 'wfs', 'sample_size')
SUBJECTIVE_SCALE_HINTS = (
    'likert',
    'expert rating',
    'expert score',
    'specialist rating',
    'mean score',
    'median score',
    'mean compatibility score',
    'quality score',
    'gqs',
    'discern',
    '5-point',
    '5 point',
    '6-point',
    '6 point',
    '7-point',
    '7 point',
    'ordinal',
    'score',
    'rating',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare family-first meta-analysis candidate input')
    parser.add_argument('--input', required=True, help='Extraction CSV input path')
    parser.add_argument('--output', required=True, help='Prepared meta-input CSV output path')
    parser.add_argument(
        '--families',
        default='accuracy,sensitivity,specificity,auc_auroc,f1',
        help='Comma-separated allowed metric families',
    )
    parser.add_argument('--default-n', type=int, default=100, help='Default sample size when not reported')
    parser.add_argument(
        '--tier-input',
        default='',
        help='Optional tier-labeled CSV for tier/crl/wfs enrichment; if omitted, auto-detected',
    )
    parser.add_argument(
        '--candidate-universe-input',
        default='',
        help='Optional locked candidate-universe CSV (record_id); if omitted, auto-detected when available',
    )
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
        normalized = normalized[:-1]
        parsed = safe_float(normalized)
        return parsed / 100.0 if parsed is not None else None
    try:
        return float(normalized)
    except ValueError:
        return None


def parse_json_like(payload: str) -> dict[str, Any] | None:
    if not payload:
        return None
    candidate = payload.strip()
    if not candidate or candidate.lower() in {'none', 'not_reported', 'nan'}:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(candidate)
        except Exception:
            return None
    return parsed if isinstance(parsed, dict) else None


def normalize_metric_text(metric_raw: str) -> str:
    lowered = metric_raw.lower().strip()
    lowered = re.sub(r'\s+', ' ', lowered)
    return lowered


def first_author_from_authors(authors_raw: str) -> str:
    text = (authors_raw or '').strip()
    if not text:
        return 'Not reported'
    lead_block = re.split(r';| and |\|', text, maxsplit=1)[0].strip()
    lead_block = re.sub(r'\([^)]*\)', '', lead_block).strip()
    if not lead_block:
        return 'Not reported'
    if ',' in lead_block:
        surname = lead_block.split(',', 1)[0].strip()
        return surname or 'Not reported'
    tokens = [token.strip(' ,.') for token in lead_block.split() if token.strip(' ,.')]
    if len(tokens) >= 2 and tokens[-2].lower() in {'da', 'de', 'del', 'den', 'der', 'di', 'du', 'la', 'le', 'van', 'von'}:
        return f'{tokens[-2]} {tokens[-1]}'
    return tokens[-1] if tokens else 'Not reported'


def classify_metric_family(metric_raw: str) -> tuple[str | None, str]:
    metric_text = normalize_metric_text(metric_raw)
    for family, patterns in METRIC_PATTERNS:
        if any(pattern in metric_text for pattern in patterns):
            return family, metric_text
    return None, metric_text


def normalize_proportion(raw_value: float) -> tuple[float | None, str | None]:
    value = raw_value
    rule = None
    if value > 1.0:
        if value <= 100.0:
            value = value / 100.0
            rule = 'percent_to_proportion'
        else:
            return None, 'out_of_range_gt_100'
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
    return value, rule


def is_subjective_scale_accuracy(
    family: str,
    metric_text: str,
    outcome: dict[str, Any],
    row: dict[str, str],
    raw_value: float,
) -> bool:
    if family != 'accuracy':
        return False
    if raw_value <= 1.0 or raw_value > 10.0:
        return False
    haystacks = [
        metric_text,
        normalize_metric_text(str(outcome.get('metric') or '')),
        normalize_metric_text(str(row.get('title') or '')),
        normalize_metric_text(str(row.get('abstract') or '')),
        normalize_metric_text(str(outcome.get('label') or '')),
        normalize_metric_text(str(outcome.get('notes') or '')),
    ]
    combined = ' '.join(part for part in haystacks if part)
    return any(hint in combined for hint in SUBJECTIVE_SCALE_HINTS)


def infer_gi_subsite(row: dict[str, str]) -> str:
    existing = (row.get('gi_subsite') or '').strip()
    if existing:
        return existing
    text = f"{row.get('title', '')} {row.get('abstract', '')}".lower()
    for subsite, keywords in GI_SUBSITE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return subsite
    return 'general_gi'


def infer_tier_input_path(input_path: pathlib.Path) -> pathlib.Path | None:
    extraction_dir = input_path.parent
    candidates = [
        extraction_dir.parent / 'tier_labeled' / 'llm_tier_labels.csv',
        extraction_dir.parent / 'tier_labeled' / 'llm_tier_labels_audited.csv',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def infer_candidate_universe_path(input_path: pathlib.Path) -> pathlib.Path | None:
    extraction_dir = input_path.parent
    candidates = [
        extraction_dir.parent / 'meta' / 'meta_input_audited.csv',
        extraction_dir.parent / 'meta' / 'meta_input_real.csv',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_candidate_ids(candidate_path: pathlib.Path | None) -> set[str]:
    if candidate_path is None or not candidate_path.exists():
        return set()
    with candidate_path.open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    output: set[str] = set()
    for row in rows:
        record_id = (row.get('record_id') or '').strip()
        if record_id:
            output.add(record_id)
    return output


def load_tier_lookup(tier_input: pathlib.Path | None) -> dict[str, dict[str, str]]:
    if tier_input is None or not tier_input.exists():
        return {}
    with tier_input.open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        record_id = (row.get('record_id') or '').strip()
        if not record_id:
            continue
        lookup[record_id] = {field: (row.get(field) or '').strip() for field in TIER_FIELDS}
    return lookup


def extract_sample_size(
    row: dict[str, str],
    outcome: dict[str, Any],
    tier_info: dict[str, str],
    default_n: int,
) -> tuple[int, bool]:
    for key in ('sample_size_total', 'sample_size', 'n', 'total_n'):
        parsed = safe_float(row.get(key))
        if parsed is not None and parsed >= 10:
            return int(round(parsed)), False
    for key in ('sample_size', 'n', 'total_n', 'denominator', 'patients', 'cases'):
        parsed = safe_float(outcome.get(key))
        if parsed is not None and parsed >= 10:
            return int(round(parsed)), False
    tier_sample_size = safe_float(tier_info.get('sample_size'))
    if tier_sample_size is not None and tier_sample_size >= 10:
        return int(round(tier_sample_size)), False
    return int(default_n), True


def compute_binomial_se(value: float, sample_size: int) -> float:
    sample = max(sample_size, 1)
    return math.sqrt(max(value * (1.0 - value), 0.0) / sample)


def main() -> None:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    allowed_families = {item.strip() for item in args.families.split(',') if item.strip()}

    tier_input_path = pathlib.Path(args.tier_input) if args.tier_input else infer_tier_input_path(input_path)
    tier_lookup = load_tier_lookup(tier_input_path)
    candidate_universe_path = (
        pathlib.Path(args.candidate_universe_input)
        if args.candidate_universe_input
        else infer_candidate_universe_path(input_path)
    )
    locked_candidate_ids = load_candidate_ids(candidate_universe_path)

    with input_path.open('r', encoding='utf-8-sig', newline='') as handle:
        source_rows = list(csv.DictReader(handle))

    output_rows: list[dict[str, str]] = []
    skipped = Counter()
    conversion_notes = Counter()

    for row in source_rows:
        decision = (row.get('decision') or '').strip().lower()
        if decision and decision != 'include':
            continue

        outcome = parse_json_like(row.get('outcome_primary', ''))
        if outcome is None:
            skipped['no_outcome_primary'] += 1
            continue

        metric_raw = str(outcome.get('metric') or row.get('metric') or '').strip()
        if not metric_raw:
            skipped['missing_metric'] += 1
            continue
        family, metric_normalized = classify_metric_family(metric_raw)
        if family is None:
            skipped['metric_not_poolable'] += 1
            continue
        if family not in allowed_families:
            skipped[f'family_filtered_{family}'] += 1
            continue

        value_raw = safe_float(outcome.get('value'))
        if value_raw is None:
            skipped['missing_value'] += 1
            continue
        if is_subjective_scale_accuracy(family, metric_normalized, outcome, row, value_raw):
            skipped['subjective_scale_accuracy_not_meta_compatible'] += 1
            continue
        value, normalization_rule = normalize_proportion(value_raw)
        if value is None:
            skipped[f'invalid_value_{normalization_rule or "unknown"}'] += 1
            continue
        if normalization_rule:
            conversion_notes[normalization_rule] += 1

        record_id = (row.get('record_id') or '').strip()
        if locked_candidate_ids and record_id and record_id not in locked_candidate_ids:
            skipped['not_in_locked_candidate_universe'] += 1
            continue
        tier_info = tier_lookup.get(record_id, {})
        sample_size, sample_size_imputed = extract_sample_size(row, outcome, tier_info, args.default_n)
        se_value = compute_binomial_se(value, sample_size)
        tier_value = tier_info.get('tier', '').strip() or (row.get('tier') or '').strip()
        crl_value = tier_info.get('crl', '').strip() or (row.get('crl') or '').strip()
        wfs_value = tier_info.get('wfs', '').strip() or (row.get('wfs') or '').strip()
        output_rows.append(
            {
                'record_id': record_id,
                'study_label': (row.get('title') or 'Untitled study')[:140],
                'first_author': first_author_from_authors(str(row.get('authors') or '')),
                'effect_size': f'{value:.6f}',
                'se': f'{se_value:.6f}',
                'metric': metric_normalized,
                'metric_family': family,
                'sample_size': str(sample_size),
                'sample_size_imputed': 'true' if sample_size_imputed else 'false',
                'gi_subsite': infer_gi_subsite(row),
                'tier': tier_value,
                'crl': crl_value,
                'wfs': wfs_value,
                'publication_year': (row.get('publication_year') or '').strip(),
                'journal_or_source': (row.get('journal_or_source') or '').strip(),
                'source_database': (row.get('source_database') or '').strip(),
                'decision': row.get('decision', ''),
                'candidate_universe': 'true',
            }
        )

    output_rows.sort(key=lambda item: (item['metric_family'], item['record_id']))

    fields = [
        'record_id',
        'study_label',
        'first_author',
        'effect_size',
        'se',
        'metric',
        'metric_family',
        'sample_size',
        'sample_size_imputed',
        'gi_subsite',
        'tier',
        'crl',
        'wfs',
        'publication_year',
        'journal_or_source',
        'source_database',
        'decision',
        'candidate_universe',
    ]
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    summary_payload = {
        'input': str(input_path),
        'output': str(output_path),
        'tier_input': str(tier_input_path) if tier_input_path else None,
        'candidate_universe_input': str(candidate_universe_path) if candidate_universe_path else None,
        'locked_candidate_count': len(locked_candidate_ids),
        'poolable_records': len(output_rows),
        'family_distribution': dict(Counter(row['metric_family'] for row in output_rows)),
        'metric_distribution': dict(Counter(row['metric'] for row in output_rows)),
        'sample_size_imputed': sum(row['sample_size_imputed'] == 'true' for row in output_rows),
        'skipped_reasons': dict(skipped),
        'normalization_notes': dict(conversion_notes),
    }
    summary_path = output_path.with_name(output_path.stem + '_summary.json')
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(f"[prepare_meta_input] poolable records: {summary_payload['poolable_records']}")
    print(f"[prepare_meta_input] family distribution: {summary_payload['family_distribution']}")
    print(f"[prepare_meta_input] imputed sample size: {summary_payload['sample_size_imputed']}")
    print(f"[prepare_meta_input] skipped: {summary_payload['skipped_reasons']}")
    print(f'[prepare_meta_input] output: {output_path}')
    print(f'[prepare_meta_input] summary: {summary_path}')


if __name__ == '__main__':
    main()
