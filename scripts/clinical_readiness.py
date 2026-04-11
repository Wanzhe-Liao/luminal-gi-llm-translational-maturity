from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
from collections import Counter, defaultdict
from typing import Any

from build_evidence_map import (
    SimpleCanvas,
    draw_label,
    draw_title_block,
    first_nonempty,
    generate_demo_rows,
    parse_wfs,
    save_canvas_outputs,
    write_csv,
)
from pipeline_lib import CRL_LEVELS, TIERS, WFS_ORDER

HIGH_EVIDENCE_TIERS = {'S', 'I-a', 'I-b'}
READINESS_STAGE_ORDER = [
    'not_ready',
    'human_review_only',
    'external_validation_needed',
    'prospective_trial_candidate',
]
READINESS_STAGE_SET = set(READINESS_STAGE_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build clinical-readiness outputs from tier-labeled GI oncology LLM studies.')
    parser.add_argument('--input', help='Input tier-label CSV')
    parser.add_argument('--output-dir', required=True, help='Directory for readiness outputs')
    parser.add_argument('--study-validation-n', type=int, default=30, help='Study-level validation sample size target')
    parser.add_argument('--use-case-validation-n', type=int, default=40, help='Use-case-level validation sample size target')
    parser.add_argument('--validation-seed', type=int, default=42, help='Random seed for reproducible validation sampling')
    parser.add_argument('--validated-use-cases', help='Optional adjudicated full use-case validation CSV (97/97)')
    parser.add_argument('--validated-agreement-summary', help='Optional adjudicated agreement summary CSV')
    parser.add_argument('--validated-methods-note', help='Optional adjudicated readiness methods note markdown')
    parser.add_argument(
        '--require-adjudicated-validation',
        action='store_true',
        help='Fail unless the full adjudicated 97/97 readiness validation inputs are present and loaded.',
    )
    parser.add_argument('--demo', action='store_true', help='Generate demo data and run full readiness analysis')
    return parser.parse_args()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def normalize_records(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in raw_rows:
        tier = first_nonempty(row, 'tier')
        crl = first_nonempty(row, 'crl')
        if tier not in TIERS or crl not in CRL_LEVELS:
            continue
        wfs = [item for item in parse_wfs(first_nonempty(row, 'wfs')) if item in WFS_ORDER]
        if not wfs:
            wfs = ['research_qc']
        records.append({
            'record_id': first_nonempty(row, 'record_id'),
            'title': first_nonempty(row, 'title', default='Untitled study'),
            'tier': tier,
            'crl': crl,
            'wfs': wfs,
            'wfs_primary': wfs[0],
            'gi_subsite': first_nonempty(row, 'gi_subsite', default='general_gi'),
            'sample_size': first_nonempty(row, 'sample_size', default='not_reported'),
            'year': first_nonempty(row, 'publication_year', 'year', default='not_reported'),
        })
    return records


def build_demo_rows() -> list[dict[str, Any]]:
    random.seed(42)
    rows = generate_demo_rows(50)
    for index, row in enumerate(rows[:12]):
        row['crl'] = 'High'
        row['tier'] = random.choice(['I-a', 'I-b', 'II'])
        row['wfs'] = 'treatment|mdt' if index % 2 == 0 else 'diagnosis|treatment'
    return rows


def readiness_stage(record: dict[str, Any], wfs_label: str) -> tuple[str, str]:
    crl = record['crl']
    tier = record['tier']
    if crl == 'High' and tier in {'S', 'I-a'}:
        return 'prospective_trial_candidate', 'High-risk use with stronger real-world evidence should move toward prospective or trial-grade validation.'
    if crl == 'High' and tier == 'I-b':
        return 'external_validation_needed', 'High-risk retrospective evidence needs external or multicenter validation before prospective escalation.'
    if crl == 'Medium' and tier in HIGH_EVIDENCE_TIERS:
        return 'external_validation_needed', 'Medium-risk real-world evidence should next undergo external validation.'
    if crl == 'Low' and tier in {'I-a', 'I-b'}:
        return 'human_review_only', 'Low-risk use cases supported by real-world evidence may be considered only within tightly supervised human-review workflows.'
    return 'not_ready', 'Current evidence remains insufficient for translational deployment and should remain in pre-validation development.'


def overlay_tags(record: dict[str, Any], wfs_label: str) -> list[str]:
    tags: list[str] = []
    tier = record['tier']
    crl = record['crl']
    if tier in {'II', 'III'}:
        tags.append('needs_real_world_validation')
    if tier in {'I-a', 'I-b'} and crl in {'Medium', 'High'}:
        tags.append('needs_external_multicenter_validation')
    if crl == 'High':
        tags.append('needs_safety_audit')
    if crl == 'High' and (wfs_label in {'treatment', 'mdt'} or tier in {'I-a', 'I-b'}):
        tags.append('rct_priority')
    if crl == 'High' and wfs_label == 'patient_communication':
        tags.append('patient_facing_high_risk')
    return tags


def classify_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for wfs_label in record['wfs']:
            stage, rationale = readiness_stage(record, wfs_label)
            tags = overlay_tags(record, wfs_label)
            rows.append({
                'record_id': record['record_id'],
                'title': record['title'],
                'wfs': wfs_label,
                'task_name': wfs_label,
                'tier': record['tier'],
                'crl': record['crl'],
                'readiness_stage': stage,
                'readiness_class': stage,
                'overlay_tags': '|'.join(tags),
                'rationale': rationale,
            })
    return rows


def rct_design_suggestion(wfs_label: str) -> tuple[str, str, str]:
    if wfs_label == 'treatment':
        return ('Patient-level superiority RCT', 'Treatment-plan concordance and safety-adjusted decision quality', '200-500')
    if wfs_label == 'mdt':
        return ('Cluster or stepped-wedge MDT trial', 'Change in MDT recommendation quality and time-to-decision', '150-400')
    if wfs_label in {'diagnosis', 'screening', 'staging'}:
        return ('Prospective comparative effectiveness trial', 'Diagnostic accuracy or management-impact endpoint', '250-600')
    return ('Prospective pragmatic trial', 'Workflow efficiency with patient-safety guardrails', '100-300')


def build_rct_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if record['tier'] not in {'I-a', 'I-b'} or record['crl'] != 'High':
            continue
        primary_wfs = 'treatment' if 'treatment' in record['wfs'] else record['wfs'][0]
        design, endpoint, sample_range = rct_design_suggestion(primary_wfs)
        rows.append({
            'record_id': record['record_id'],
            'use_case_name': f"{primary_wfs}: {record['title'][:80]}",
            'current_evidence_grade': record['tier'],
            'suggested_rct_design': design,
            'primary_endpoint_suggestion': endpoint,
            'expected_sample_size_range': sample_range,
            'overlay_tags': '|'.join(sorted(set(overlay_tags(record, primary_wfs) + ['rct_priority']))),
        })
    return rows


def build_gap_matrix(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for wfs_label in WFS_ORDER:
        for crl in CRL_LEVELS:
            subset = [record for record in records if wfs_label in record['wfs'] and record['crl'] == crl]
            tier_s = sum(1 for record in subset if record['tier'] == 'S')
            tier_i = sum(1 for record in subset if record['tier'] in {'I-a', 'I-b'})
            tier_ii = sum(1 for record in subset if record['tier'] == 'II')
            tier_iii = sum(1 for record in subset if record['tier'] == 'III')
            critical_gap = crl == 'High' and (tier_s + tier_i == 0)
            rows.append({
                'wfs': wfs_label,
                'crl': crl,
                'tier_s_count': tier_s,
                'tier_i_count': tier_i,
                'tier_ii_count': tier_ii,
                'tier_iii_count': tier_iii,
                'total_count': len(subset),
                'critical_gap': 'yes' if critical_gap else 'no',
            })
    return rows


def stratified_round_robin_sample(records: list[dict[str, Any]], target_n: int, seed: int, key_fields: list[str]) -> list[dict[str, Any]]:
    if target_n <= 0 or not records:
        return []
    rng = random.Random(seed)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = tuple(str(record.get(field, 'unknown')) for field in key_fields)
        groups[key].append(record)
    ordered_keys = list(groups.keys())
    rng.shuffle(ordered_keys)
    for key in ordered_keys:
        rng.shuffle(groups[key])
    sampled: list[dict[str, Any]] = []
    while ordered_keys and len(sampled) < target_n:
        next_round: list[tuple[str, ...]] = []
        for key in ordered_keys:
            bucket = groups[key]
            if bucket and len(sampled) < target_n:
                sampled.append(bucket.pop())
            if bucket:
                next_round.append(key)
        ordered_keys = next_round
    return sampled


def build_study_level_validation_sample(records: list[dict[str, Any]], sample_n: int, seed: int) -> list[dict[str, Any]]:
    target = min(max(sample_n, 0), len(records))
    sampled = stratified_round_robin_sample(
        records,
        target,
        seed,
        ['tier', 'crl', 'gi_subsite', 'wfs_primary'],
    )
    rows: list[dict[str, Any]] = []
    for row in sampled:
        rows.append({
            'record_id': row['record_id'],
            'title': row['title'],
            'year': row['year'],
            'tier': row['tier'],
            'crl': row['crl'],
            'gi_subsite': row['gi_subsite'],
            'wfs_primary': row['wfs_primary'],
            'wfs_all': '|'.join(row['wfs']),
            'sample_size': row['sample_size'],
            'validation_stratum': f"{row['tier']}|{row['crl']}|{row['gi_subsite']}|{row['wfs_primary']}",
            'selection_method': 'stratified_round_robin',
            'reviewer_1': '',
            'reviewer_2': '',
            'adjudicator': '',
            'review_status': 'pending',
            'notes': '',
        })
    return rows


def build_use_case_validation_sample(readiness_rows: list[dict[str, Any]], sample_n: int, seed: int) -> list[dict[str, Any]]:
    target = min(max(sample_n, 0), len(readiness_rows))
    sampled = stratified_round_robin_sample(
        readiness_rows,
        target,
        seed,
        ['readiness_stage', 'tier', 'crl', 'wfs'],
    )
    rows: list[dict[str, Any]] = []
    for row in sampled:
        rows.append({
            'record_id': row['record_id'],
            'title': row['title'],
            'wfs': row['wfs'],
            'task_name': row['task_name'],
            'tier': row['tier'],
            'crl': row['crl'],
            'readiness_stage': row['readiness_stage'],
            'overlay_tags': row['overlay_tags'],
            'validation_stratum': f"{row['readiness_stage']}|{row['tier']}|{row['crl']}|{row['wfs']}",
            'selection_method': 'stratified_round_robin',
            'reviewer_1': '',
            'reviewer_2': '',
            'adjudicator': '',
            'review_status': 'pending',
            'notes': '',
        })
    return rows


def render_gap_matrix(base_path: pathlib.Path, gap_rows: list[dict[str, Any]]) -> None:
    width = 1460
    height = 920
    canvas = SimpleCanvas(width, height)
    draw_title_block(canvas, 'Clinical readiness gap matrix', 'High-risk cells without Tier S/I evidence are flagged as critical gaps.')
    left = 210
    top = 140
    cell_w = 240
    cell_h = 150
    max_total = max((int(row['total_count']) for row in gap_rows), default=1)
    for row_index, wfs_label in enumerate(WFS_ORDER):
        draw_label(canvas, 20, top + row_index * 68 + 16, wfs_label.replace('_', ' '), scale=1)
    for col_index, crl in enumerate(CRL_LEVELS):
        draw_label(canvas, left + col_index * cell_w + 50, 95, crl, scale=2)
    for row_index, wfs_label in enumerate(WFS_ORDER):
        for col_index, crl in enumerate(CRL_LEVELS):
            row = next(item for item in gap_rows if item['wfs'] == wfs_label and item['crl'] == crl)
            x = left + col_index * cell_w
            y = top + row_index * 68
            total = int(row['total_count'])
            shade = 255 - int((total / max_total) * 160) if max_total else 255
            fill = (255, max(80, shade), max(80, shade)) if row['critical_gap'] == 'yes' else (shade, shade, 255)
            canvas.fill_rect(x, y, cell_w - 20, 56, fill)
            canvas.draw_rect(x, y, cell_w - 20, 56, (40, 40, 40))
            counts = f"S:{row['tier_s_count']} I:{row['tier_i_count']} II:{row['tier_ii_count']} III:{row['tier_iii_count']}"
            draw_label(canvas, x + 10, y + 8, counts, scale=1)
            draw_label(canvas, x + 10, y + 30, f"N={total}", scale=1)
    save_canvas_outputs(canvas, base_path)


def canonicalize_role_name(role: str) -> str:
    normalized = str(role).strip().lower()
    if normalized.startswith('reviewer 1') or normalized in {'reviewer_1', 'r1'}:
        return 'Reviewer A'
    if normalized.startswith('reviewer 2') or normalized in {'reviewer_2', 'r2'}:
        return 'Reviewer B'
    if 'adjudicator' in normalized or normalized in {'arbiter', 'tie-breaker', 'tiebreaker'}:
        return 'Adjudicator'
    return role.strip() or 'Reviewer'


def load_validated_use_case_rows(path: pathlib.Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return [], []

    sanitized_rows: list[dict[str, Any]] = []
    readiness_rows: list[dict[str, Any]] = []
    for row in rows:
        final_stage = first_nonempty(row, 'adjudicated_final_stage', 'readiness_stage', default='not_ready')
        final_stage = final_stage if final_stage in READINESS_STAGE_SET else 'not_ready'
        sanitized = dict(row)
        sanitized['reviewer_1_name'] = 'Reviewer A'
        sanitized['reviewer_2_name'] = 'Reviewer B'
        sanitized['adjudicator_name'] = 'Adjudicator'
        sanitized['readiness_stage'] = final_stage
        sanitized['readiness_class'] = final_stage
        sanitized_rows.append(sanitized)
        readiness_rows.append(
            {
                'record_id': first_nonempty(row, 'record_id'),
                'title': first_nonempty(row, 'title', default='Untitled study'),
                'wfs': first_nonempty(row, 'wfs', default='research_qc'),
                'task_name': first_nonempty(row, 'task_name', default=first_nonempty(row, 'wfs', default='research_qc')),
                'tier': first_nonempty(row, 'tier', default='II'),
                'crl': first_nonempty(row, 'crl', default='Medium'),
                'readiness_stage': final_stage,
                'readiness_class': final_stage,
                'overlay_tags': first_nonempty(row, 'overlay_tags'),
                'rationale': first_nonempty(row, 'adjudication_reason', 'rationale'),
            }
        )
    return sanitized_rows, readiness_rows


def load_validation_agreement(path: pathlib.Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else {}


def ensure_adjudicated_validation_inputs(
    *,
    require_adjudicated_validation: bool,
    validated_use_case_path: pathlib.Path,
    validated_agreement_path: pathlib.Path,
    validated_methods_note_path: pathlib.Path,
) -> None:
    if not require_adjudicated_validation:
        return
    missing = [
        str(path)
        for path in [
            validated_use_case_path,
            validated_agreement_path,
            validated_methods_note_path,
        ]
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "submission-clean readiness rebuild requires adjudicated validation inputs; missing: "
            + ", ".join(missing)
        )


def main() -> None:
    args = parse_args()
    if not args.demo and not args.input:
        raise SystemExit('--input is required unless --demo is used')

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        raw_rows = build_demo_rows()
    else:
        import csv
        with open(args.input, 'r', encoding='utf-8-sig', newline='') as handle:
            raw_rows = list(csv.DictReader(handle))

    records = normalize_records(raw_rows)
    if not records:
        raise SystemExit('No valid tier-labeled records found')

    readiness_rows = classify_records(records)
    readiness_validation_rows: list[dict[str, Any]] = []
    agreement_row: dict[str, Any] = {}
    validation_status = 'pending_dual_review'

    validated_use_case_path = pathlib.Path(args.validated_use_cases) if args.validated_use_cases else (output_dir / 'readiness_validation_full_97_2026-04-06.csv')
    validated_agreement_path = pathlib.Path(args.validated_agreement_summary) if args.validated_agreement_summary else (output_dir / 'readiness_validation_agreement_summary_2026-04-06.csv')
    validated_methods_note_path = pathlib.Path(args.validated_methods_note) if args.validated_methods_note else (output_dir / 'readiness_validation_methods_note_2026-04-06.md')
    ensure_adjudicated_validation_inputs(
        require_adjudicated_validation=args.require_adjudicated_validation,
        validated_use_case_path=validated_use_case_path,
        validated_agreement_path=validated_agreement_path,
        validated_methods_note_path=validated_methods_note_path,
    )

    if validated_use_case_path.exists():
        readiness_validation_rows, adjudicated_readiness_rows = load_validated_use_case_rows(validated_use_case_path)
        if adjudicated_readiness_rows:
            readiness_rows = adjudicated_readiness_rows
            validation_status = 'completed_dual_review_with_adjudication'
    if validated_agreement_path.exists():
        agreement_row = load_validation_agreement(validated_agreement_path)
    if validated_methods_note_path.exists() and validated_methods_note_path.resolve() != (output_dir / validated_methods_note_path.name).resolve():
        (output_dir / validated_methods_note_path.name).write_text(validated_methods_note_path.read_text(encoding='utf-8'), encoding='utf-8')
    if args.require_adjudicated_validation:
        if validation_status != 'completed_dual_review_with_adjudication':
            raise SystemExit('submission-clean readiness rebuild requires completed dual review with adjudication')
        if not agreement_row:
            raise SystemExit('submission-clean readiness rebuild requires an adjudicated agreement summary row')
    records_by_id = {record['record_id']: record for record in records}
    rct_rows = build_rct_candidates(records)
    if validation_status == 'completed_dual_review_with_adjudication':
        adjudicated_rct_rows = []
        seen_rct_ids: set[str] = set()
        for row in readiness_rows:
            if row.get('readiness_stage') != 'prospective_trial_candidate':
                continue
            record_id = str(row.get('record_id', '')).strip()
            if not record_id or record_id in seen_rct_ids:
                continue
            seen_rct_ids.add(record_id)
            source_record = records_by_id.get(record_id, {})
            wfs_label = str(row.get('wfs') or row.get('task_name') or source_record.get('wfs_primary') or 'other').strip()
            primary_wfs = 'treatment' if 'treatment' in wfs_label else wfs_label.split('|')[0]
            design, endpoint, sample_range = rct_design_suggestion(primary_wfs)
            overlay_value = str(row.get('overlay_tags') or '').strip()
            overlay_parts = [part.strip() for part in overlay_value.split('|') if part.strip()]
            if 'rct_priority' not in overlay_parts:
                overlay_parts.append('rct_priority')
            adjudicated_rct_rows.append({
                'record_id': record_id,
                'use_case_name': f"{primary_wfs}: {str(row.get('title') or source_record.get('title') or 'Untitled study')[:80]}",
                'current_evidence_grade': str(source_record.get('tier') or row.get('tier') or 'not_reported'),
                'suggested_rct_design': design,
                'primary_endpoint_suggestion': endpoint,
                'expected_sample_size_range': sample_range,
                'overlay_tags': '|'.join(sorted(set(overlay_parts))),
            })
        rct_rows = adjudicated_rct_rows
    gap_rows = build_gap_matrix(records)

    readiness_path = output_dir / 'readiness_classification.csv'
    shortlist_path = output_dir / 'readiness_shortlist.csv'
    rct_path = output_dir / 'rct_candidates.csv'
    gap_path = output_dir / 'gap_matrix.csv'
    validation_study_path = output_dir / 'validation_sample_study_level.csv'
    validation_use_case_path = output_dir / 'validation_sample_use_case_level.csv'
    heatmap_base = output_dir / 'gap_matrix_heatmap'
    report_path = output_dir / 'clinical_readiness_report.json'

    write_csv(readiness_path, ['record_id', 'title', 'wfs', 'task_name', 'tier', 'crl', 'readiness_stage', 'readiness_class', 'overlay_tags', 'rationale'], readiness_rows)
    shortlisted_rows = [row for row in readiness_rows if row['readiness_stage'] in {'external_validation_needed', 'prospective_trial_candidate'}]
    write_csv(shortlist_path, ['record_id', 'title', 'wfs', 'task_name', 'tier', 'crl', 'readiness_stage', 'readiness_class', 'overlay_tags', 'rationale'], shortlisted_rows)
    write_csv(rct_path, ['record_id', 'use_case_name', 'current_evidence_grade', 'suggested_rct_design', 'primary_endpoint_suggestion', 'expected_sample_size_range', 'overlay_tags'], rct_rows)
    write_csv(gap_path, ['wfs', 'crl', 'tier_s_count', 'tier_i_count', 'tier_ii_count', 'tier_iii_count', 'total_count', 'critical_gap'], gap_rows)
    study_validation_rows = build_study_level_validation_sample(records, args.study_validation_n, args.validation_seed)
    use_case_validation_rows = build_use_case_validation_sample(readiness_rows, args.use_case_validation_n, args.validation_seed + 1)
    write_csv(
        validation_study_path,
        ['record_id', 'title', 'year', 'tier', 'crl', 'gi_subsite', 'wfs_primary', 'wfs_all', 'sample_size', 'validation_stratum', 'selection_method', 'reviewer_1', 'reviewer_2', 'adjudicator', 'review_status', 'notes'],
        study_validation_rows,
    )
    write_csv(
        validation_use_case_path,
        ['record_id', 'title', 'wfs', 'task_name', 'tier', 'crl', 'readiness_stage', 'overlay_tags', 'validation_stratum', 'selection_method', 'reviewer_1', 'reviewer_2', 'adjudicator', 'review_status', 'notes'],
        use_case_validation_rows,
    )
    if readiness_validation_rows:
        validated_path = output_dir / 'readiness_validation_full_97_2026-04-06.csv'
        write_csv(
            validated_path,
            [
                'record_id',
                'title',
                'wfs',
                'task_name',
                'tier',
                'crl',
                'readiness_stage',
                'readiness_class',
                'overlay_tags',
                'rationale',
                'reviewer_1_name',
                'reviewer_1_stage',
                'reviewer_1_notes',
                'reviewer_2_name',
                'reviewer_2_stage',
                'reviewer_2_notes',
                'adjudicator_name',
                'adjudicated_final_stage',
                'adjudication_reason',
                'pre_adjudication_disagreement_flag',
                'safety_escalation_flag',
                'source_classification_revisit_flag',
                'source_classification_revisit_note',
            ],
            readiness_validation_rows,
        )
    if agreement_row:
        agreement_path = output_dir / 'readiness_validation_agreement_summary_2026-04-06.csv'
        write_csv(agreement_path, list(agreement_row.keys()), [agreement_row])
    render_gap_matrix(heatmap_base, gap_rows)

    if validation_status == 'completed_dual_review_with_adjudication' and agreement_row:
        validation_sampling_payload: dict[str, Any] = {
            'study_level_target_n': len(records),
            'study_level_sampled_n': len(records),
            'use_case_level_target_n': safe_int(agreement_row.get('total_use_cases', len(readiness_rows)), len(readiness_rows)),
            'use_case_level_sampled_n': safe_int(agreement_row.get('adjudicated_completed_n', len(readiness_rows)), len(readiness_rows)),
            'seed': args.validation_seed,
            'study_level_strata_covered': len({f"{row['tier']}|{row['crl']}|{row['gi_subsite']}|{row['wfs_primary']}" for row in records}),
            'use_case_level_strata_covered': len({f"{row.get('readiness_stage')}|{row.get('tier')}|{row.get('crl')}|{row.get('wfs')}" for row in readiness_rows}),
            'pre_adjudication_concordant_n': safe_int(agreement_row.get('pre_adjudication_concordant_n')),
            'pre_adjudication_discordant_n': safe_int(agreement_row.get('pre_adjudication_discordant_n')),
            'pre_adjudication_percent_agreement': safe_float(agreement_row.get('pre_adjudication_percent_agreement')),
            'cohens_kappa': safe_float(agreement_row.get('cohens_kappa')),
            'kappa_method': agreement_row.get('kappa_method', ''),
        }
    else:
        validation_sampling_payload = {
            'study_level_target_n': args.study_validation_n,
            'study_level_sampled_n': len(study_validation_rows),
            'use_case_level_target_n': args.use_case_validation_n,
            'use_case_level_sampled_n': len(use_case_validation_rows),
            'seed': args.validation_seed,
            'study_level_strata_covered': len({row['validation_stratum'] for row in study_validation_rows}),
            'use_case_level_strata_covered': len({row['validation_stratum'] for row in use_case_validation_rows}),
        }

    readiness_counter = Counter(row['readiness_stage'] for row in readiness_rows)
    report = {
        'record_count': len(records),
        'use_case_count': len(readiness_rows),
        'readiness_stage_basis': (
            'use_case_level_adjudicated_validation'
            if validation_status == 'completed_dual_review_with_adjudication'
            else 'use_case_level'
        ),
        'validation_status': validation_status,
        'readiness_counts': dict(readiness_counter),
        'overlay_tag_counts': dict(Counter(tag for row in readiness_rows for tag in row['overlay_tags'].split('|') if tag)),
        'rct_candidate_count': (
            int(readiness_counter.get('prospective_trial_candidate', 0))
            if validation_status == 'completed_dual_review_with_adjudication'
            else len(rct_rows)
        ),
        'critical_gap_count': sum(1 for row in gap_rows if row['critical_gap'] == 'yes'),
        'validation_sampling': validation_sampling_payload,
        'paths': {
            'readiness_classification': str(readiness_path),
            'readiness_shortlist': str(shortlist_path),
            'rct_candidates': str(rct_path),
            'gap_matrix': str(gap_path),
            'validation_sample_study_level': str(validation_study_path),
            'validation_sample_use_case_level': str(validation_use_case_path),
            'gap_matrix_heatmap_png': str(heatmap_base.with_suffix('.png')),
            'gap_matrix_heatmap_pdf': str(heatmap_base.with_suffix('.pdf')),
        },
    }
    if validation_status == 'completed_dual_review_with_adjudication':
        report['paths']['readiness_validation_full'] = str(validated_use_case_path)
        report['paths']['readiness_validation_agreement_summary'] = str(validated_agreement_path)
        report['paths']['readiness_validation_methods_note'] = str(validated_methods_note_path)
        if agreement_row:
            report['agreement_summary'] = agreement_row
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(f'[clinical_readiness] readiness={readiness_path}')
    print(f'[clinical_readiness] readiness_shortlist={shortlist_path}')
    print(f'[clinical_readiness] rct_candidates={rct_path}')
    print(f'[clinical_readiness] gap_matrix={gap_path}')
    print(f'[clinical_readiness] validation_study={validation_study_path}')
    print(f'[clinical_readiness] validation_use_case={validation_use_case_path}')
    print(f'[clinical_readiness] report={report_path}')


if __name__ == '__main__':
    main()
