from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import re
from collections import Counter, defaultdict
from statistics import median
from typing import Any

from build_evidence_map import first_nonempty, generate_demo_rows, write_csv
from pipeline_lib import CRL_LEVELS, TIERS, WFS_ORDER

DEFAULT_METADATA = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'protocol-a-live-2026-03-06' / 'protocol_a_master_unique.csv'
REVIEW_LIKE_TOKENS = (
    'review',
    'systematic review',
    'meta-analysis',
    'meta analysis',
    'scoping review',
    'narrative review',
    'editorial',
    'comment',
    'guideline',
    'consensus',
    'position statement',
)
ABSTRACT_ONLY_TOKENS = (
    'conference abstract',
    'meeting abstract',
    'poster abstract',
    'abstracts',
)
SHORT_FORMAT_TOKENS = (
    'conference paper',
    'proceedings paper',
    'proceeding paper',
    'letter',
    'research letter',
    'brief report',
    'short communication',
    'rapid communication',
)
PRIMARY_EVIDENCE_TOKENS = (
    'journal article',
    'article',
    'clinical trial',
    'comparative study',
    'evaluation study',
    'validation study',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Audit publication form labels and sensitivity impact for included studies.')
    parser.add_argument('--included-csv', help='Included studies CSV (typically tier_labeled/llm_tier_labels.csv)')
    parser.add_argument('--metadata-csv', default=str(DEFAULT_METADATA), help='Upstream metadata CSV with publication_type and peer_review_status')
    parser.add_argument('--tripod-csv', help='TRIPOD scoring CSV (tripod/tripod_scores.csv)')
    parser.add_argument('--readiness-csv', help='Readiness classification CSV (readiness/readiness_classification.csv)')
    parser.add_argument('--output-dir', required=True, help='Directory for publication-form audit outputs')
    parser.add_argument(
        '--sensitivity-exclude-classes',
        default='abstract_only_insufficient',
        help='Comma-separated publication_form_class values excluded for sensitivity analysis',
    )
    parser.add_argument('--demo', action='store_true', help='Use generated demo rows')
    return parser.parse_args()


def load_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def normalize_doi(value: str) -> str:
    text = (value or '').strip().lower()
    if not text:
        return ''
    text = text.replace('https://doi.org/', '').replace('http://doi.org/', '')
    text = text.replace('doi:', '').strip()
    return text


def normalize_title(value: str) -> str:
    text = (value or '').lower()
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    return ' '.join(text.split())


def classify_publication_form(
    *,
    publication_type: str,
    peer_review_status: str,
    title: str,
    abstract: str,
) -> tuple[str, str]:
    pub_text = (publication_type or '').lower()
    title_text = (title or '').lower()
    has_primary_article = any(token in pub_text for token in PRIMARY_EVIDENCE_TOKENS)
    has_abstract_only = any(token in pub_text for token in ABSTRACT_ONLY_TOKENS)
    has_short_format = any(token in pub_text for token in SHORT_FORMAT_TOKENS)
    has_review_like = any(token in pub_text for token in REVIEW_LIKE_TOKENS) or any(
        token in title_text for token in ('systematic review', 'scoping review', 'meta-analysis', 'meta analysis')
    )

    if has_review_like:
        return 'review_like_non_original', 'Publication metadata indicates review/editorial/comment style content.'
    if has_abstract_only and not has_primary_article:
        return 'abstract_only_insufficient', 'Publication type is conference/meeting abstract without full-article detail.'
    if has_short_format and not has_primary_article:
        return 'original_data_short_format', 'Short-format source (conference paper/letter/brief format) retained as supplementary primary evidence.'
    if has_primary_article:
        return 'data_rich_primary_evidence', 'Publication type reflects article-level original evidence.'
    if has_short_format:
        return 'original_data_short_format', 'Short-format source retained because publication metadata suggests original data format.'
    if (peer_review_status or '').strip().lower() == 'peer_reviewed' and not pub_text:
        return 'data_rich_primary_evidence', 'Peer-reviewed status present with missing publication_type; classified as primary evidence.'
    return 'unknown', 'Metadata was insufficient for deterministic publication-form class assignment.'


def match_metadata(
    row: dict[str, str],
    *,
    by_doi: dict[str, list[dict[str, str]]],
    by_title: dict[str, list[dict[str, str]]],
    by_record: dict[str, dict[str, str]],
) -> tuple[dict[str, str] | None, str]:
    doi = normalize_doi(first_nonempty(row, 'doi'))
    if doi and doi in by_doi:
        candidates = by_doi[doi]
        if len(candidates) == 1:
            return candidates[0], 'doi_exact'
        record_id = first_nonempty(row, 'record_id')
        for candidate in candidates:
            if first_nonempty(candidate, 'record_id') == record_id:
                return candidate, 'doi_exact'
        title_norm = normalize_title(first_nonempty(row, 'title'))
        for candidate in candidates:
            if normalize_title(first_nonempty(candidate, 'title')) == title_norm:
                return candidate, 'doi_exact'
        return candidates[0], 'doi_exact_ambiguous'

    title_norm = normalize_title(first_nonempty(row, 'title'))
    if title_norm and title_norm in by_title:
        candidates = by_title[title_norm]
        if len(candidates) == 1:
            return candidates[0], 'title_norm'
        record_id = first_nonempty(row, 'record_id')
        for candidate in candidates:
            if first_nonempty(candidate, 'record_id') == record_id:
                return candidate, 'title_norm'
        return candidates[0], 'title_norm_ambiguous'

    record_id = first_nonempty(row, 'record_id')
    if record_id and record_id in by_record:
        return by_record[record_id], 'record_id_exact'
    return None, 'unmatched'


def try_float(value: Any) -> float | None:
    text = str(value or '').strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def summarize_tier_stability(included_rows: list[dict[str, str]], retained_ids: set[str], full_ids: set[str]) -> dict[str, dict[str, float]]:
    full_counts = Counter(first_nonempty(row, 'tier', default='unknown') for row in included_rows if first_nonempty(row, 'record_id') in full_ids)
    retained_counts = Counter(first_nonempty(row, 'tier', default='unknown') for row in included_rows if first_nonempty(row, 'record_id') in retained_ids)
    total_full = max(len(full_ids), 1)
    total_retained = max(len(retained_ids), 1)
    output: dict[str, dict[str, float]] = {}
    for tier in list(TIERS) + sorted(set(full_counts.keys()) | set(retained_counts.keys()) - set(TIERS)):
        output[tier] = {
            'full_count': float(full_counts.get(tier, 0)),
            'full_percent': round(full_counts.get(tier, 0) * 100 / total_full, 2),
            'retained_count': float(retained_counts.get(tier, 0)),
            'retained_percent': round(retained_counts.get(tier, 0) * 100 / total_retained, 2),
            'delta_count': float(retained_counts.get(tier, 0) - full_counts.get(tier, 0)),
        }
    return output


def summarize_tripod_stability(tripod_rows: list[dict[str, str]], retained_ids: set[str], full_ids: set[str]) -> dict[str, Any]:
    scores_full = [try_float(first_nonempty(row, 'total_score')) for row in tripod_rows if first_nonempty(row, 'record_id') in full_ids]
    scores_retained = [try_float(first_nonempty(row, 'total_score')) for row in tripod_rows if first_nonempty(row, 'record_id') in retained_ids]
    rates_full = [try_float(first_nonempty(row, 'compliance_rate')) for row in tripod_rows if first_nonempty(row, 'record_id') in full_ids]
    rates_retained = [try_float(first_nonempty(row, 'compliance_rate')) for row in tripod_rows if first_nonempty(row, 'record_id') in retained_ids]
    scores_full_clean = [value for value in scores_full if value is not None]
    scores_retained_clean = [value for value in scores_retained if value is not None]
    rates_full_clean = [value for value in rates_full if value is not None]
    rates_retained_clean = [value for value in rates_retained if value is not None]
    return {
        'full_n': len(scores_full_clean),
        'retained_n': len(scores_retained_clean),
        'full_mean_total_score': round(sum(scores_full_clean) / len(scores_full_clean), 3) if scores_full_clean else None,
        'retained_mean_total_score': round(sum(scores_retained_clean) / len(scores_retained_clean), 3) if scores_retained_clean else None,
        'full_median_total_score': round(median(scores_full_clean), 3) if scores_full_clean else None,
        'retained_median_total_score': round(median(scores_retained_clean), 3) if scores_retained_clean else None,
        'full_mean_compliance_rate': round(sum(rates_full_clean) / len(rates_full_clean), 3) if rates_full_clean else None,
        'retained_mean_compliance_rate': round(sum(rates_retained_clean) / len(rates_retained_clean), 3) if rates_retained_clean else None,
    }


def summarize_readiness_stability(readiness_rows: list[dict[str, str]], retained_ids: set[str], full_ids: set[str]) -> dict[str, Any]:
    full_rows = [row for row in readiness_rows if first_nonempty(row, 'record_id') in full_ids]
    retained_rows = [row for row in readiness_rows if first_nonempty(row, 'record_id') in retained_ids]
    full_counts = Counter(first_nonempty(row, 'readiness_stage', default='unknown') for row in full_rows)
    retained_counts = Counter(first_nonempty(row, 'readiness_stage', default='unknown') for row in retained_rows)
    total_full = max(len(full_rows), 1)
    total_retained = max(len(retained_rows), 1)
    stages = sorted(set(full_counts.keys()) | set(retained_counts.keys()))
    stage_distribution = {}
    for stage in stages:
        stage_distribution[stage] = {
            'full_count': full_counts.get(stage, 0),
            'full_percent': round(full_counts.get(stage, 0) * 100 / total_full, 2),
            'retained_count': retained_counts.get(stage, 0),
            'retained_percent': round(retained_counts.get(stage, 0) * 100 / total_retained, 2),
        }
    return {
        'full_use_case_n': len(full_rows),
        'retained_use_case_n': len(retained_rows),
        'readiness_stage_distribution': stage_distribution,
    }


def build_sensitivity_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append({
        'domain': 'study_counts',
        'metric': 'full_study_count',
        'full_value': summary.get('full_study_count', 0),
        'retained_value': summary.get('retained_study_count', 0),
        'delta': summary.get('retained_study_count', 0) - summary.get('full_study_count', 0),
    })
    tier_block = summary.get('tier_distribution', {})
    for tier, metrics in tier_block.items():
        rows.append({
            'domain': 'tier_distribution',
            'metric': f'{tier}_count',
            'full_value': metrics.get('full_count', 0),
            'retained_value': metrics.get('retained_count', 0),
            'delta': metrics.get('retained_count', 0) - metrics.get('full_count', 0),
        })
        rows.append({
            'domain': 'tier_distribution',
            'metric': f'{tier}_percent',
            'full_value': metrics.get('full_percent', 0),
            'retained_value': metrics.get('retained_percent', 0),
            'delta': round(metrics.get('retained_percent', 0) - metrics.get('full_percent', 0), 2),
        })

    tripod = summary.get('tripod_summary', {})
    for metric in [
        'full_mean_total_score',
        'retained_mean_total_score',
        'full_median_total_score',
        'retained_median_total_score',
        'full_mean_compliance_rate',
        'retained_mean_compliance_rate',
    ]:
        rows.append({
            'domain': 'tripod',
            'metric': metric,
            'full_value': tripod.get(metric) if metric.startswith('full_') else tripod.get(metric.replace('retained_', 'full_')),
            'retained_value': tripod.get(metric) if metric.startswith('retained_') else tripod.get(metric.replace('full_', 'retained_')),
            'delta': '',
        })

    readiness = summary.get('readiness_summary', {}).get('readiness_stage_distribution', {})
    for stage, metrics in readiness.items():
        rows.append({
            'domain': 'readiness_stage_distribution',
            'metric': f'{stage}_count',
            'full_value': metrics.get('full_count', 0),
            'retained_value': metrics.get('retained_count', 0),
            'delta': metrics.get('retained_count', 0) - metrics.get('full_count', 0),
        })
        rows.append({
            'domain': 'readiness_stage_distribution',
            'metric': f'{stage}_percent',
            'full_value': metrics.get('full_percent', 0),
            'retained_value': metrics.get('retained_percent', 0),
            'delta': round(metrics.get('retained_percent', 0) - metrics.get('full_percent', 0), 2),
        })
    return rows


def build_demo_rows_for_audit() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    random.seed(42)
    base = generate_demo_rows(20)
    included_rows: list[dict[str, str]] = []
    metadata_rows: list[dict[str, str]] = []
    tripod_rows: list[dict[str, str]] = []
    readiness_rows: list[dict[str, str]] = []
    publication_types = [
        'Journal Article',
        'Conference Abstract',
        'Conference paper',
        'Letter',
        'Editorial',
        'Article',
    ]
    readiness_stages = ['not_ready', 'human_review_only', 'external_validation_needed', 'prospective_trial_candidate']
    for idx, row in enumerate(base, start=1):
        record_id = row.get('record_id', f'demo_{idx:03d}')
        doi = f'10.1000/demo.{idx:03d}'
        publication_type = publication_types[idx % len(publication_types)]
        included_rows.append({
            'record_id': record_id,
            'title': row.get('title', f'Demo study {idx}'),
            'abstract': row.get('abstract', ''),
            'doi': doi,
            'tier': row.get('tier', 'II'),
            'crl': row.get('crl', CRL_LEVELS[idx % len(CRL_LEVELS)]),
            'wfs': row.get('wfs', WFS_ORDER[idx % len(WFS_ORDER)]),
        })
        metadata_rows.append({
            'record_id': record_id,
            'title': row.get('title', f'Demo study {idx}'),
            'doi': doi,
            'publication_type': publication_type,
            'peer_review_status': 'peer_reviewed',
        })
        tripod_rows.append({
            'record_id': record_id,
            'total_score': str(5 + idx % 10),
            'compliance_rate': str(round((5 + idx % 10) / 19 * 100, 2)),
        })
        readiness_rows.append({
            'record_id': record_id,
            'readiness_stage': readiness_stages[idx % len(readiness_stages)],
        })
    return included_rows, metadata_rows, tripod_rows, readiness_rows


def main() -> None:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        included_rows, metadata_rows, tripod_rows, readiness_rows = build_demo_rows_for_audit()
    else:
        if not args.included_csv:
            raise SystemExit('--included-csv is required unless --demo is used')
        included_rows = load_csv(pathlib.Path(args.included_csv))
        metadata_rows = load_csv(pathlib.Path(args.metadata_csv))
        tripod_rows = load_csv(pathlib.Path(args.tripod_csv)) if args.tripod_csv and pathlib.Path(args.tripod_csv).exists() else []
        readiness_rows = load_csv(pathlib.Path(args.readiness_csv)) if args.readiness_csv and pathlib.Path(args.readiness_csv).exists() else []

    by_doi: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_title: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_record: dict[str, dict[str, str]] = {}
    for row in metadata_rows:
        doi = normalize_doi(first_nonempty(row, 'doi'))
        if doi:
            by_doi[doi].append(row)
        title_norm = normalize_title(first_nonempty(row, 'title'))
        if title_norm:
            by_title[title_norm].append(row)
        record_id = first_nonempty(row, 'record_id')
        if record_id:
            by_record[record_id] = row

    audit_rows: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()
    method_counter: Counter[str] = Counter()
    for row in included_rows:
        matched, method = match_metadata(row, by_doi=by_doi, by_title=by_title, by_record=by_record)
        source = matched or {}
        publication_type = first_nonempty(source, 'publication_type')
        peer_review_status = first_nonempty(source, 'peer_review_status')
        publication_form_class, class_reason = classify_publication_form(
            publication_type=publication_type,
            peer_review_status=peer_review_status,
            title=first_nonempty(row, 'title'),
            abstract=first_nonempty(row, 'abstract'),
        )
        method_counter[method] += 1
        class_counter[publication_form_class] += 1
        audit_rows.append({
            'record_id': first_nonempty(row, 'record_id'),
            'title': first_nonempty(row, 'title'),
            'doi': first_nonempty(row, 'doi'),
            'tier': first_nonempty(row, 'tier'),
            'crl': first_nonempty(row, 'crl'),
            'wfs': first_nonempty(row, 'wfs'),
            'source_database': first_nonempty(row, 'source_database'),
            'publication_type': publication_type,
            'peer_review_status': peer_review_status,
            'metadata_match_method': method,
            'metadata_match_status': 'matched' if method != 'unmatched' else 'unmatched',
            'publication_form_class': publication_form_class,
            'classification_reason': class_reason,
        })

    excluded_classes = {item.strip() for item in args.sensitivity_exclude_classes.split(',') if item.strip()}
    full_ids = {first_nonempty(row, 'record_id') for row in audit_rows if first_nonempty(row, 'record_id')}
    retained_ids = {
        first_nonempty(row, 'record_id')
        for row in audit_rows
        if first_nonempty(row, 'record_id') and first_nonempty(row, 'publication_form_class') not in excluded_classes
    }

    tier_distribution = summarize_tier_stability(included_rows, retained_ids, full_ids)
    tripod_summary = summarize_tripod_stability(tripod_rows, retained_ids, full_ids)
    readiness_summary = summarize_readiness_stability(readiness_rows, retained_ids, full_ids)

    sensitivity_summary = {
        'full_study_count': len(full_ids),
        'retained_study_count': len(retained_ids),
        'removed_study_count': len(full_ids - retained_ids),
        'removed_record_ids': sorted(full_ids - retained_ids),
        'excluded_classes': sorted(excluded_classes),
        'tier_distribution': tier_distribution,
        'tripod_summary': tripod_summary,
        'readiness_summary': readiness_summary,
    }
    sensitivity_rows = build_sensitivity_rows(sensitivity_summary)

    summary = {
        'included_study_count': len(audit_rows),
        'matched_count': sum(1 for row in audit_rows if row['metadata_match_status'] == 'matched'),
        'unmatched_count': sum(1 for row in audit_rows if row['metadata_match_status'] == 'unmatched'),
        'metadata_match_method_counts': dict(method_counter),
        'publication_form_class_counts': dict(class_counter),
        'output_files': {
            'publication_form_audit_csv': str(output_dir / 'publication_form_audit.csv'),
            'publication_form_summary_json': str(output_dir / 'publication_form_summary.json'),
            'publication_form_sensitivity_summary_json': str(output_dir / 'publication_form_sensitivity_summary.json'),
            'publication_form_sensitivity_table_csv': str(output_dir / 'publication_form_sensitivity_table.csv'),
        },
    }

    write_csv(
        output_dir / 'publication_form_audit.csv',
        [
            'record_id',
            'title',
            'doi',
            'tier',
            'crl',
            'wfs',
            'source_database',
            'publication_type',
            'peer_review_status',
            'metadata_match_method',
            'metadata_match_status',
            'publication_form_class',
            'classification_reason',
        ],
        audit_rows,
    )
    write_csv(
        output_dir / 'publication_form_sensitivity_table.csv',
        ['domain', 'metric', 'full_value', 'retained_value', 'delta'],
        sensitivity_rows,
    )
    (output_dir / 'publication_form_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output_dir / 'publication_form_sensitivity_summary.json').write_text(json.dumps(sensitivity_summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(f"[publication_form_audit] audit={output_dir / 'publication_form_audit.csv'}")
    print(f"[publication_form_audit] summary={output_dir / 'publication_form_summary.json'}")
    print(f"[publication_form_audit] sensitivity={output_dir / 'publication_form_sensitivity_summary.json'}")


if __name__ == '__main__':
    main()
