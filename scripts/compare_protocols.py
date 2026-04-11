from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
from collections import Counter, defaultdict
from datetime import date
from typing import Any

from build_evidence_map import SimpleCanvas, first_nonempty, generate_demo_rows, write_csv
from pipeline_lib import CRL_LEVELS, TIERS

WS = re.compile(r'\s+')
MANUAL_VERIFICATION_PLACEHOLDER = 'pending_manual_spotcheck'
SUPPLEMENTARY_TABLE_S6 = 'table_s6_source_review_and_inclusion_rationale_summary.csv'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare Protocol A and Protocol B tier-labeled datasets with maturation-oriented outputs.')
    parser.add_argument('--protocol-a', help='Protocol A tier-labeled CSV')
    parser.add_argument('--protocol-b', help='Protocol B tier-labeled CSV')
    parser.add_argument('--protocol-a-tripod', help='Optional Protocol A tripod_scores.csv')
    parser.add_argument('--protocol-b-tripod', help='Optional Protocol B tripod_scores.csv')
    parser.add_argument('--publication-form-audit', help='Optional explicit publication_form_audit.csv used for overlap audit alignment during internal comparison rebuilds')
    parser.add_argument('--output-dir', required=True, help='Output directory for comparison results')
    parser.add_argument('--demo', action='store_true', help='Generate demo Protocol A/B datasets and run comparison')
    return parser.parse_args()


def load_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def normalize_text(value: str) -> str:
    return WS.sub(' ', re.sub(r'[^a-z0-9]+', ' ', (value or '').lower())).strip()


def normalize_doi(value: str) -> str:
    return normalize_text(value).replace('https doi org ', '').replace('http doi org ', '')


def parse_year(row: dict[str, str]) -> int:
    raw = first_nonempty(row, 'publication_year', 'year', default='0')
    try:
        return int(float(raw))
    except Exception:
        return 0


def parse_date_like(row: dict[str, str]) -> tuple[date | None, str]:
    for key in ('publication_date', 'date', 'published', 'statusVerifiedDate'):
        raw = first_nonempty(row, key)
        if not raw:
            continue
        match = re.match(r'(\d{4})-(\d{2})-(\d{2})', raw)
        if match:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3))), 'day'
    year = parse_year(row)
    return (date(year, 1, 1), 'year') if year else (None, 'none')


def build_key(row: dict[str, str]) -> str:
    doi = normalize_doi(first_nonempty(row, 'doi'))
    if doi:
        return f'doi:{doi}'
    title = normalize_text(first_nonempty(row, 'title'))
    if title:
        return f'title:{title}'
    return f"record:{first_nonempty(row, 'record_id', default='unknown')}"


def discover_tripod_path(protocol_path: pathlib.Path, explicit: str | None) -> pathlib.Path | None:
    if explicit:
        path = pathlib.Path(explicit)
        return path if path.exists() else None
    base = protocol_path.parent.parent
    candidate = base / 'tripod' / 'tripod_scores.csv'
    return candidate if candidate.exists() else None


def discover_publication_form_audit_path(protocol_path: pathlib.Path) -> pathlib.Path | None:
    base = protocol_path.parent.parent
    candidates = [
        base / 'supplementary' / SUPPLEMENTARY_TABLE_S6,
        base / 'supplementary' / 'table_s7_publication_form_audit.csv',
        base / 'publication_form' / 'publication_form_audit.csv',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def discover_publication_form_summary_path(protocol_path: pathlib.Path) -> pathlib.Path | None:
    base = protocol_path.parent.parent
    candidate = base / 'publication_form' / 'publication_form_sensitivity_summary.json'
    return candidate if candidate.exists() else None


def load_json(path: pathlib.Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def load_publication_form_audit_records(path: pathlib.Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for index, row in enumerate(load_csv(path), start=1):
        title = first_nonempty(row, 'title', default='Untitled')
        doi = first_nonempty(row, 'doi', default='')
        records.append({
            'record_id': first_nonempty(row, 'record_id', default=f'audit_{index:04d}'),
            'title': title,
            'doi': doi,
            'doi_norm': normalize_doi(doi),
            'tier': '',
            'crl': '',
            'source': 'peer_reviewed_audit',
            'year': 0,
            'date': None,
            'date_precision': 'none',
            'compliance_rate': None,
            'title_norm': normalize_text(title),
            'key': f'audit:{index:04d}',
            'raw': row,
        })
    return records


def classify_publication_status_group(source: str) -> str:
    normalized = normalize_text(source)
    if any(token in normalized for token in ('medrxiv', 'biorxiv', 'arxiv', 'preprint')):
        return 'preprint'
    if any(token in normalized for token in ('clinicaltrials', 'trial registry', 'registry')):
        return 'registry'
    if any(token in normalized for token in ('embase', 'google scholar', 'indexed')):
        return 'indexed_non_journal_source'
    return 'other_source_stream'


def tier_counts_from_publication_form_summary(summary: dict[str, Any]) -> Counter[str]:
    tier_distribution = summary.get('tier_distribution') or {}
    counts: Counter[str] = Counter()
    for tier in TIERS:
        raw_count = (tier_distribution.get(tier) or {}).get('full_count')
        if raw_count is None:
            continue
        counts[tier] = int(round(float(raw_count)))
    return counts


def load_tripod_by_record(path: pathlib.Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    return {row.get('record_id', '').strip(): row for row in load_csv(path) if row.get('record_id', '').strip()}


def load_records(path: pathlib.Path, tripod_path: pathlib.Path | None = None) -> list[dict[str, Any]]:
    rows = load_csv(path)
    tripod_by_id = load_tripod_by_record(tripod_path)
    unique_by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        tier = first_nonempty(row, 'tier')
        if tier not in TIERS:
            continue
        record_id = first_nonempty(row, 'record_id', default='')
        doi_norm = normalize_doi(first_nonempty(row, 'doi'))
        title_norm = normalize_text(first_nonempty(row, 'title'))
        tripod_row = tripod_by_id.get(record_id, {})
        compliance_raw = first_nonempty(row, 'compliance_rate', default='') or first_nonempty(tripod_row, 'compliance_rate', default='')
        try:
            compliance_rate = float(compliance_raw) if compliance_raw not in {'', 'error'} else None
        except Exception:
            compliance_rate = None
        parsed_date, date_precision = parse_date_like(row)
        record = {
            'record_id': record_id,
            'title': first_nonempty(row, 'title', default='Untitled'),
            'doi': first_nonempty(row, 'doi', default=''),
            'doi_norm': doi_norm,
            'tier': tier,
            'crl': first_nonempty(row, 'crl', default='Low'),
            'source': first_nonempty(row, 'source_database', 'source', 'journal_or_source', default='unknown'),
            'year': parse_year(row),
            'date': parsed_date,
            'date_precision': date_precision,
            'compliance_rate': compliance_rate,
            'title_norm': title_norm,
            'key': build_key(row),
            'raw': row,
        }
        unique_by_key.setdefault(record['key'], record)
    return list(unique_by_key.values())


def gammaincc(a: float, x: float) -> float:
    if a <= 0 or x < 0:
        return 1.0
    if x == 0:
        return 1.0
    eps = 1e-14
    fpmin = 1e-300
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(1, 500):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * eps:
                break
        gamser = total * math.exp(-x + a * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - gamser))
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / max(b, fpmin)
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return max(0.0, min(1.0, math.exp(-x + a * math.log(x) - gln) * h))


def chi_square_test(counts_a: dict[str, int], counts_b: dict[str, int], categories: list[str]) -> dict[str, Any]:
    total_a = sum(counts_a.get(cat, 0) for cat in categories)
    total_b = sum(counts_b.get(cat, 0) for cat in categories)
    grand = total_a + total_b
    if total_a == 0 or total_b == 0 or grand == 0:
        return {'chi_square': 0.0, 'df': 0, 'p_value': 1.0}
    chi = 0.0
    used = 0
    for cat in categories:
        observed_a = counts_a.get(cat, 0)
        observed_b = counts_b.get(cat, 0)
        cat_total = observed_a + observed_b
        if cat_total == 0:
            continue
        expected_a = total_a * cat_total / grand
        expected_b = total_b * cat_total / grand
        if expected_a > 0:
            chi += (observed_a - expected_a) ** 2 / expected_a
        if expected_b > 0:
            chi += (observed_b - expected_b) ** 2 / expected_b
        used += 1
    df = max(0, used - 1)
    p = gammaincc(df / 2.0, chi / 2.0) if df > 0 else 1.0
    return {'chi_square': round(chi, 4), 'df': df, 'p_value': round(p, 6)}


def select_best_partner(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    def rank(record: dict[str, Any]) -> tuple[int, int, int, str]:
        date_ord = record['date'].toordinal() if record['date'] else 0
        return (1 if record['doi_norm'] else 0, date_ord, record['year'], record['record_id'])

    return max(candidates, key=rank)


def match_records(
    records_a: list[dict[str, Any]],
    records_b: list[dict[str, Any]],
    match_scope: str = 'primary_set',
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    doi_index_a: dict[str, list[dict[str, Any]]] = defaultdict(list)
    title_index_a: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records_a:
        if record['doi_norm']:
            doi_index_a[record['doi_norm']].append(record)
        if record['title_norm']:
            title_index_a[record['title_norm']].append(record)

    matched: list[dict[str, Any]] = []
    unique_b: list[dict[str, Any]] = []
    used_a_ids: set[str] = set()
    match_type_counts: Counter[str] = Counter()

    for record_b in records_b:
        partner = None
        match_type = ''

        if record_b['doi_norm']:
            doi_candidates = [record for record in doi_index_a.get(record_b['doi_norm'], []) if record['record_id'] not in used_a_ids]
            if doi_candidates:
                partner = select_best_partner(doi_candidates)
                match_type = 'doi_exact'

        if partner is None and record_b['title_norm']:
            title_candidates = [record for record in title_index_a.get(record_b['title_norm'], []) if record['record_id'] not in used_a_ids]
            if title_candidates:
                partner = select_best_partner(title_candidates)
                if record_b['doi_norm'] and partner['doi_norm'] and record_b['doi_norm'] != partner['doi_norm']:
                    match_type = 'title_exact_cross_doi'
                else:
                    match_type = 'title_exact'

        if partner is None:
            unique_b.append(record_b)
            continue

        used_a_ids.add(partner['record_id'])
        match_type_counts[match_type] += 1
        matched.append({
            'record_a': partner,
            'record_b': record_b,
            'match_type': match_type,
            'match_scope': match_scope,
            'manually_verified': MANUAL_VERIFICATION_PLACEHOLDER,
        })

    return matched, unique_b, dict(match_type_counts)


def overlap_summary(records_a: list[dict[str, Any]], matched_pairs: list[dict[str, Any]], unique_b: list[dict[str, Any]]) -> dict[str, int]:
    return {
        'protocol_a_only': max(len(records_a) - len(matched_pairs), 0),
        'protocol_b_only': len(unique_b),
        'overlap': len(matched_pairs),
    }


def render_grouped_bars(path: pathlib.Path, title: str, categories: list[str], counts_a: dict[str, int], counts_b: dict[str, int], label_a: str, label_b: str) -> None:
    canvas = SimpleCanvas(1400, 860)
    canvas.draw_text(30, 30, title[:55].upper(), (0, 51, 102), scale=2)
    left, top, width, height = 100, 130, 1120, 580
    canvas.draw_line(left, top, left, top + height, (30, 30, 30))
    canvas.draw_line(left, top + height, left + width, top + height, (30, 30, 30))
    max_value = max([counts_a.get(cat, 0) for cat in categories] + [counts_b.get(cat, 0) for cat in categories] + [1])
    group_width = width / max(1, len(categories))
    for idx, cat in enumerate(categories):
        x = left + int(idx * group_width + 20)
        a_height = int(height * counts_a.get(cat, 0) / max_value)
        b_height = int(height * counts_b.get(cat, 0) / max_value)
        canvas.fill_rect(x, top + height - a_height, 36, a_height, (0, 51, 102))
        canvas.fill_rect(x + 44, top + height - b_height, 36, b_height, (204, 0, 0))
        canvas.draw_text(x, top + height + 16, cat[:10], (40, 40, 40), scale=1)
    canvas.draw_text(1240, 150, label_a[:16], (0, 51, 102), scale=1)
    canvas.draw_text(1240, 180, label_b[:16], (204, 0, 0), scale=1)
    canvas.save_png(path)


def render_source_distribution(path: pathlib.Path, source_counts: Counter[str]) -> None:
    canvas = SimpleCanvas(1300, 820)
    canvas.draw_text(30, 30, 'PROTOCOL B SOURCE DISTRIBUTION', (0, 51, 102), scale=2)
    left, top, width, height = 240, 120, 920, 560
    canvas.draw_line(left, top, left, top + height, (30, 30, 30))
    canvas.draw_line(left, top + height, left + width, top + height, (30, 30, 30))
    items = source_counts.most_common()
    max_value = max((count for _, count in items), default=1)
    bar_h = max(30, height // max(1, len(items) + 1))
    colors = [(0, 51, 102), (0, 128, 128), (106, 81, 163), (230, 126, 34), (102, 102, 102)]
    for idx, (source, count) in enumerate(items):
        y = top + idx * bar_h + 16
        length = int(width * count / max_value)
        canvas.draw_text(20, y, source[:22], (40, 40, 40), scale=1)
        canvas.fill_rect(left, y, length, bar_h - 10, colors[idx % len(colors)])
        canvas.draw_text(left + length + 10, y, str(count), (40, 40, 40), scale=1)
    canvas.save_png(path)


def render_time_lag(path: pathlib.Path, lags: list[int]) -> None:
    canvas = SimpleCanvas(1300, 820)
    canvas.draw_text(30, 30, 'PREPRINT TO PUBLISHED TIME LAG', (0, 51, 102), scale=2)
    left, top, width, height = 100, 130, 1120, 560
    canvas.draw_line(left, top, left, top + height, (30, 30, 30))
    canvas.draw_line(left, top + height, left + width, top + height, (30, 30, 30))
    if not lags:
        canvas.draw_text(200, 320, 'NO MATCHED TIME-LAG PAIRS', (102, 102, 102), scale=2)
        canvas.save_png(path)
        return
    bins = [0, 90, 180, 270, 365, 730]
    labels = ['0-89', '90-179', '180-269', '270-364', '365-729', '730+']
    counts = [0] * len(labels)
    for lag in lags:
        placed = False
        for idx in range(len(bins) - 1):
            if bins[idx] <= lag < bins[idx + 1]:
                counts[idx] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    max_value = max(counts) or 1
    group_width = width / len(labels)
    for idx, label in enumerate(labels):
        x = left + int(idx * group_width + 20)
        h = int(height * counts[idx] / max_value)
        canvas.fill_rect(x, top + height - h, 70, h, (204, 0, 0))
        canvas.draw_text(x, top + height + 16, label, (40, 40, 40), scale=1)
        canvas.draw_text(x + 16, top + height - h - 18, str(counts[idx]), (0, 51, 102), scale=1)
    canvas.save_png(path)


def build_submission_facing_summary(summary: dict[str, Any]) -> dict[str, Any]:
    protocol_b_total = int(summary.get('protocol_b_total') or 0)
    protocol_b_unique = int(summary.get('protocol_b_unique_records') or 0)
    overlap = int(((summary.get('overlap') or {}).get('overlap')) or 0)
    protocol_a_total = int(summary.get('protocol_a_total') or 0)
    return {
        'status': 'submission_facing_descriptive_summary',
        'analysis_scope': 'supplementary publication-status descriptive comparison only',
        'descriptive_comparison_only': True,
        'internal_authority_source': 'protocol_comparison/comparison_summary.json',
        'expanded_peer_reviewed_comparison_set_count': protocol_a_total,
        'supplementary_publication_status_records_reviewed': protocol_b_total,
        'supplementary_publication_status_records_unique': protocol_b_unique,
        'supplementary_publication_status_records_overlap': overlap,
        'supplementary_publication_status_signature': f'{protocol_b_total} reviewed / {protocol_b_unique} unique / {overlap} overlap',
        'overlap_review_level': 'concept_level',
    }


def build_demo_inputs(output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    protocol_a = output_dir / 'demo_protocol_a.csv'
    protocol_b = output_dir / 'demo_protocol_b.csv'
    rows = generate_demo_rows(40)
    rows_a = []
    rows_b = []
    b_sources = ['medrxiv', 'arxiv', 'clinicaltrials', 'google scholar']
    for idx, row in enumerate(rows, start=1):
        record = {
            'record_id': row.get('record_id', f'rec_{idx:03d}'),
            'title': row.get('title', f'Demo {idx}'),
            'doi': f'10.1000/demo.{idx:03d}' if idx <= 12 else '',
            'publication_year': str(row.get('publication_year', 2024)),
            'tier': row.get('tier', 'II'),
            'crl': row.get('crl', 'Medium'),
            'source_database': 'pubmed' if idx % 2 else 'scopus',
            'compliance_rate': str(round(55 + (idx % 30), 1)),
        }
        rows_a.append(record)
        if idx <= 10 or idx >= 25:
            record_b = dict(record)
            record_b['source_database'] = b_sources[idx % len(b_sources)]
            if idx <= 10:
                record_b['publication_year'] = str(max(2022, int(record_b['publication_year']) - 1))
            if idx >= 30:
                record_b['doi'] = ''
                record_b['title'] = f"Protocol B unique preprint {idx}: {record_b['title']}"
            rows_b.append(record_b)
    write_csv(protocol_a, list(rows_a[0].keys()), rows_a)
    write_csv(protocol_b, list(rows_b[0].keys()), rows_b)
    return protocol_a, protocol_b


def main() -> None:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        protocol_a_path, protocol_b_path = build_demo_inputs(output_dir)
        tripod_a_path = None
        tripod_b_path = None
    else:
        if not args.protocol_a or not args.protocol_b:
            raise SystemExit('--protocol-a and --protocol-b are required unless --demo is used')
        protocol_a_path = pathlib.Path(args.protocol_a)
        protocol_b_path = pathlib.Path(args.protocol_b)
        tripod_a_path = discover_tripod_path(protocol_a_path, args.protocol_a_tripod)
        tripod_b_path = discover_tripod_path(protocol_b_path, args.protocol_b_tripod)

    records_a = load_records(protocol_a_path, tripod_a_path)
    records_b = load_records(protocol_b_path, tripod_b_path)
    if args.demo:
        publication_form_audit_path = None
    elif args.publication_form_audit:
        publication_form_audit_path = pathlib.Path(args.publication_form_audit)
        if not publication_form_audit_path.exists():
            raise SystemExit(f'--publication-form-audit not found: {publication_form_audit_path}')
    else:
        publication_form_audit_path = discover_publication_form_audit_path(protocol_a_path)

    publication_form_audit_records = load_publication_form_audit_records(publication_form_audit_path)
    primary_matches, remaining_unique_b, primary_match_type_counts = match_records(records_a, records_b, match_scope='primary_set')
    audit_matches, unique_b, audit_match_type_counts = match_records(
        publication_form_audit_records,
        remaining_unique_b,
        match_scope='peer_reviewed_audit_only',
    )
    matched_pairs = primary_matches + audit_matches
    match_type_counts = Counter(primary_match_type_counts)
    match_type_counts.update(audit_match_type_counts)
    overlap = overlap_summary(records_a, matched_pairs, unique_b)

    tier_counts_a = Counter(record['tier'] for record in records_a)
    tier_counts_b = Counter(record['tier'] for record in records_b)
    crl_counts_a = Counter(record['crl'] for record in records_a if record['crl'] in CRL_LEVELS)
    crl_counts_b = Counter(record['crl'] for record in records_b if record['crl'] in CRL_LEVELS)
    b_source_counts = Counter(record['source'] for record in records_b)
    b_publication_status_counts = Counter(classify_publication_status_group(record['source']) for record in records_b)
    unique_b_publication_status_counts = Counter(classify_publication_status_group(record['source']) for record in unique_b)

    tier_test = chi_square_test(tier_counts_a, tier_counts_b, TIERS)
    crl_test = chi_square_test(crl_counts_a, crl_counts_b, CRL_LEVELS)
    valid_tripod_a = [record['compliance_rate'] for record in records_a if record['compliance_rate'] is not None]
    valid_tripod_b = [record['compliance_rate'] for record in records_b if record['compliance_rate'] is not None]
    tripod_mean_a = round(sum(valid_tripod_a) / len(valid_tripod_a), 2) if valid_tripod_a else None
    tripod_mean_b = round(sum(valid_tripod_b) / len(valid_tripod_b), 2) if valid_tripod_b else None

    lag_days = []
    maturation_rows = []
    matched_pair_rows = []
    for pair in matched_pairs:
        record_a = pair['record_a']
        record_b = pair['record_b']
        lag = None
        if record_a['date'] and record_b['date'] and record_a.get('date_precision') == 'day' and record_b.get('date_precision') == 'day':
            lag = abs((record_a['date'] - record_b['date']).days)
            lag_days.append(lag)
        matched_pair_rows.append({
            'record_id_a': record_a['record_id'],
            'record_id_b': record_b['record_id'],
            'title_a': record_a['title'],
            'title_b': record_b['title'],
            'doi_a': record_a['doi'],
            'doi_b': record_b['doi'],
            'match_type': pair['match_type'],
            'match_scope': pair['match_scope'],
            'manually_verified': pair['manually_verified'],
            'year_a': record_a['year'],
            'year_b': record_b['year'],
            'source_b': record_b['source'],
            'main_study_set_status': first_nonempty(record_a['raw'], 'main_study_set_status', default=''),
            'classification_reason': first_nonempty(record_a['raw'], 'classification_reason', default=''),
            'lag_days': lag if lag is not None else '',
            'tier_a': record_a['tier'],
            'tier_b': record_b['tier'],
            'crl_a': record_a['crl'],
            'crl_b': record_b['crl'],
            'tripod_a': record_a['compliance_rate'] if record_a['compliance_rate'] is not None else '',
            'tripod_b': record_b['compliance_rate'] if record_b['compliance_rate'] is not None else '',
        })
        maturation_rows.append({
            'record_id_a': record_a['record_id'],
            'record_id_b': record_b['record_id'],
            'key': record_a['key'],
            'title_a': record_a['title'],
            'title_b': record_b['title'],
            'doi_a': record_a['doi'],
            'doi_b': record_b['doi'],
            'match_type': pair['match_type'],
            'match_scope': pair['match_scope'],
            'manually_verified': pair['manually_verified'],
            'source_b': record_b['source'],
            'main_study_set_status': first_nonempty(record_a['raw'], 'main_study_set_status', default=''),
            'classification_reason': first_nonempty(record_a['raw'], 'classification_reason', default=''),
            'year_a': record_a['year'],
            'year_b': record_b['year'],
            'lag_days': lag if lag is not None else '',
            'tier_a': record_a['tier'],
            'tier_b': record_b['tier'],
            'crl_a': record_a['crl'],
            'crl_b': record_b['crl'],
            'tripod_a': record_a['compliance_rate'] if record_a['compliance_rate'] is not None else '',
            'tripod_b': record_b['compliance_rate'] if record_b['compliance_rate'] is not None else '',
        })

    unique_b_rows = [{
        'record_id': record['record_id'],
        'title': record['title'],
        'doi': record['doi'],
        'source': record['source'],
        'publication_status_group': classify_publication_status_group(record['source']),
        'tier': record['tier'],
        'crl': record['crl'],
        'year': record['year'],
        'compliance_rate': record['compliance_rate'] if record['compliance_rate'] is not None else '',
    } for record in unique_b]

    merged_unique = list({record['key']: record for record in records_a + unique_b}.values())
    merged_tier_counts = Counter(record['tier'] for record in merged_unique)
    sensitivity_test = chi_square_test(tier_counts_a, merged_tier_counts, TIERS)

    summary = {
        'protocol_a_total_raw': len(load_csv(protocol_a_path)),
        'protocol_b_total_raw': len(load_csv(protocol_b_path)),
        'protocol_a_total': len(records_a),
        'protocol_b_total': len(records_b),
        'protocol_b_unique_records': len(unique_b),
        'overlap': overlap,
        'match_type_counts': dict(match_type_counts),
        'source_distribution_protocol_b': dict(b_source_counts),
        'publication_status_summary': {
            'supplementary_source_records_reviewed': len(records_b),
            'unique_after_overlap_review': len(unique_b),
            'total_reviewed_counts': dict(b_publication_status_counts),
            'unique_counts': dict(unique_b_publication_status_counts),
        },
        'tier_distribution_test': tier_test,
        'crl_distribution_test': crl_test,
        'tripod_mean_protocol_a': tripod_mean_a,
        'tripod_mean_protocol_b': tripod_mean_b,
        'time_lag': {
            'matched_pairs': len(matched_pairs),
            'matched_pairs_with_precise_dates': len(lag_days),
            'median_days': sorted(lag_days)[len(lag_days) // 2] if lag_days else None,
            'all_lag_days': lag_days,
            'manual_pair_verification_status': 'pending',
            'pairwise_scope': 'descriptive_only',
        },
        'maturation_analysis': {
            'matched_pairs': len(matched_pairs),
            'tripod_delta_mean_a_minus_b': None,
            'tier_transitions_b_to_a': {},
            'crl_transitions_b_to_a': {},
            'conclusion_shift_status': 'withheld_pending_manual_verification',
            'reporting_improvement_status': 'withheld_pending_manual_verification',
            'manual_pair_verification_status': 'pending',
            'manual_pair_verification_note': 'Automated concept-level matches were exported for audit, but pairwise maturation or reporting-improvement claims were withheld pending manual spot-check.',
        },
        'sensitivity_analysis': {
            'protocol_a_tier_counts': dict(tier_counts_a),
            'merged_a_plus_b_unique_tier_counts': dict(merged_tier_counts),
            'tier_shift_test': sensitivity_test,
        },
    }

    comparison_rows = []
    for tier in TIERS:
        comparison_rows.append({'domain': 'tier', 'category': tier, 'protocol_a': tier_counts_a.get(tier, 0), 'protocol_b': tier_counts_b.get(tier, 0)})
    for crl in CRL_LEVELS:
        comparison_rows.append({'domain': 'crl', 'category': crl, 'protocol_a': crl_counts_a.get(crl, 0), 'protocol_b': crl_counts_b.get(crl, 0)})
    comparison_rows.append({'domain': 'tripod_mean', 'category': 'compliance_rate', 'protocol_a': tripod_mean_a if tripod_mean_a is not None else '', 'protocol_b': tripod_mean_b if tripod_mean_b is not None else ''})

    summary_path = output_dir / 'comparison_summary.json'
    submission_summary_path = output_dir / 'submission_facing_summary.json'
    table_path = output_dir / 'comparison_table.csv'
    tier_plot = output_dir / 'tier_comparison.png'
    source_plot = output_dir / 'source_distribution.png'
    lag_plot = output_dir / 'time_lag_distribution.png'
    matched_pairs_audit_path = output_dir / 'matched_pairs_audit.csv'
    maturation_pairs_path = output_dir / 'maturation_pairs.csv'
    unique_b_path = output_dir / 'protocol_b_unique_records.csv'

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    submission_summary_path.write_text(
        json.dumps(build_submission_facing_summary(summary), indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    write_csv(table_path, ['domain', 'category', 'protocol_a', 'protocol_b'], comparison_rows)
    write_csv(
        matched_pairs_audit_path,
        ['record_id_a', 'record_id_b', 'title_a', 'title_b', 'doi_a', 'doi_b', 'match_type', 'match_scope', 'manually_verified', 'year_a', 'year_b', 'source_b', 'main_study_set_status', 'classification_reason', 'lag_days', 'tier_a', 'tier_b', 'crl_a', 'crl_b', 'tripod_a', 'tripod_b'],
        matched_pair_rows,
    )
    write_csv(maturation_pairs_path, list(maturation_rows[0].keys()) if maturation_rows else ['record_id_a'], maturation_rows)
    write_csv(
        unique_b_path,
        list(unique_b_rows[0].keys()) if unique_b_rows else ['record_id', 'publication_status_group'],
        unique_b_rows,
    )
    render_grouped_bars(tier_plot, 'Protocol A vs Protocol B Tier Distribution', TIERS, tier_counts_a, tier_counts_b, 'Protocol A', 'Protocol B')
    render_source_distribution(source_plot, b_source_counts)
    render_time_lag(lag_plot, lag_days)

    print(json.dumps({
        'summary_json': str(summary_path),
        'submission_facing_summary_json': str(submission_summary_path),
        'comparison_table': str(table_path),
        'matched_pairs_audit': str(matched_pairs_audit_path),
        'maturation_pairs': str(maturation_pairs_path),
        'protocol_b_unique_records_csv': str(unique_b_path),
        'publication_form_audit_used': str(publication_form_audit_path) if publication_form_audit_path else '',
        'tier_plot': str(tier_plot),
        'source_plot': str(source_plot),
        'time_lag_plot': str(lag_plot),
        'protocol_b_unique_records': len(unique_b),
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
