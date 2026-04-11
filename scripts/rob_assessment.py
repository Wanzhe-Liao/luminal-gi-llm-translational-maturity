from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
from collections import Counter, defaultdict
from typing import Any

from build_evidence_map import first_nonempty, generate_demo_rows, parse_wfs, write_csv
from pipeline_lib import CRL_LEVELS, TIERS

HIGH_TIERS = {'S', 'I-a', 'I-b'}
LEVEL2_TIER_PRIORITY = {'I-a', 'I-b'}
ROB2_DOMAINS = [
    'rob2_randomization_process',
    'rob2_deviations_from_intended_interventions',
    'rob2_missing_outcome_data',
    'rob2_measurement_of_outcome',
    'rob2_selection_of_reported_result',
    'rob2_overall_bias',
    'rob2_notes',
]
ROBINS_I_DOMAINS = [
    'robins_i_confounding',
    'robins_i_selection_of_participants',
    'robins_i_classification_of_interventions',
    'robins_i_deviations_from_intended_interventions',
    'robins_i_missing_data',
    'robins_i_measurement_of_outcomes',
    'robins_i_selection_of_reported_result',
]
QUADAS2_DOMAINS = [
    'quadas2_patient_selection',
    'quadas2_index_test',
    'quadas2_reference_standard',
    'quadas2_flow_and_timing',
]
PROBAST_CORE_DOMAINS = [
    'probast_participants',
    'probast_predictors',
    'probast_outcome',
    'probast_analysis',
]
PROBAST_AI_DOMAINS = [
    'probast_ai_data_quality',
    'probast_ai_model_fairness',
    'probast_ai_validation_strategy',
    'probast_ai_real_world_applicability',
    'probast_ai_data_leakage',
    'probast_ai_overall',
]
PROBAST_DOMAINS = PROBAST_CORE_DOMAINS + PROBAST_AI_DOMAINS
ALL_DOMAIN_COLUMNS = ROB2_DOMAINS + ROBINS_I_DOMAINS + QUADAS2_DOMAINS + PROBAST_DOMAINS
AI_SIGNALS = (
    'ai', 'artificial intelligence', 'llm', 'large language model', 'gpt', 'chatgpt', 'claude',
    'gemini', 'llama', 'mistral', 'deepseek', 'transformer', 'neural', 'machine learning'
)
PREDICTION_SIGNALS = ('predict', 'prediction', 'prognostic', 'classification', 'classifier', 'risk model', 'triage')
DIAGNOSTIC_SIGNALS = ('sensitivity', 'specificity', 'auc', 'diagnosis', 'pathology', 'imaging', 'diagnostic accuracy')
OBSERVATIONAL_SIGNALS = ('prospective', 'retrospective', 'cohort', 'case-control', 'case series', 'cross-sectional')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate RoB assignment worksheets using Level-1 meta subset plus Level-2 stratified priority expansion.')
    parser.add_argument('--input', help='Input tier-label or extraction CSV')
    parser.add_argument('--meta-input', help='Meta truth CSV (expected 39-study subset for Level-1)')
    parser.add_argument('--output-dir', required=True, help='Directory for RoB outputs')
    parser.add_argument('--level2-max', type=int, default=24, help='Maximum studies added for Level-2 stratified priority expansion')
    parser.add_argument('--level2-min', type=int, default=20, help='Target minimum for Level-2 when candidate pool allows')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic stratified sampling')
    parser.add_argument('--submission-contract', action='store_true', help='Use the adjudicated submission-facing formal-assessment source instead of heuristic legacy tool assignment.')
    parser.add_argument('--formal-assessment-source', help='Submission-facing formal-assessment source CSV used when --submission-contract is set.')
    parser.add_argument('--demo', action='store_true', help='Generate demo studies and build worksheets')
    return parser.parse_args()


def normalize_record_id(value: str) -> str:
    return (value or '').strip().upper()


def load_meta_ids(path: str | None, output_dir: pathlib.Path) -> tuple[list[str], pathlib.Path | None]:
    candidate_paths: list[pathlib.Path] = []
    if path:
        candidate_paths.append(pathlib.Path(path))
    candidate_paths.extend([
        output_dir.parent / 'meta' / 'meta_input_audited.csv',
        output_dir.parent / 'meta' / 'meta_input_real.csv',
    ])
    for meta_path in candidate_paths:
        if not meta_path.exists():
            continue
        with meta_path.open('r', encoding='utf-8-sig', newline='') as handle:
            ordered_ids = []
            for row in csv.DictReader(handle):
                record_id = normalize_record_id(first_nonempty(row, 'record_id'))
                if record_id:
                    ordered_ids.append(record_id)
            deduplicated = []
            seen: set[str] = set()
            for record_id in ordered_ids:
                if record_id not in seen:
                    deduplicated.append(record_id)
                    seen.add(record_id)
            return deduplicated, meta_path
    return [], None


def build_demo_rows() -> list[dict[str, Any]]:
    random.seed(42)
    rows = generate_demo_rows(80)
    designs = ['RCT', 'retrospective cohort', 'prospective cohort', 'diagnostic_accuracy', 'prediction_model', 'classification_model']
    gi_sites = ['esophageal', 'gastric', 'colorectal', 'small_bowel', 'anal', 'general_gi']
    wfs_values = ['diagnosis', 'staging', 'treatment', 'mdt', 'followup', 'research_qc']
    for index, row in enumerate(rows):
        row['record_id'] = row.get('record_id', f'demo_{index + 1:03d}')
        row['crl'] = CRL_LEVELS[index % len(CRL_LEVELS)]
        row['tier'] = TIERS[index % len(TIERS)]
        row['study_design'] = designs[index % len(designs)]
        row['gi_subsite'] = gi_sites[index % len(gi_sites)]
        row['wfs'] = wfs_values[index % len(wfs_values)]
        row['publication_year'] = str(2022 + index % 5)
        if 'prediction' in row['study_design'] or 'classification' in row['study_design']:
            row['title'] = f"AI prediction study {index + 1}: {row['title']}"
            row['abstract'] = f"This LLM-based classification study evaluated external validation and fairness. {row.get('abstract', '')}"
    return rows


def normalize_records(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in raw_rows:
        tier = first_nonempty(row, 'tier')
        crl = first_nonempty(row, 'crl')
        record_id = normalize_record_id(first_nonempty(row, 'record_id', default=''))
        if tier not in TIERS or crl not in CRL_LEVELS or not record_id:
            continue
        wfs_values = [value for value in parse_wfs(first_nonempty(row, 'wfs')) if value]
        wfs_primary = wfs_values[0] if wfs_values else first_nonempty(row, 'wfs', default='not_reported')
        priority_flags: list[str] = []
        if tier in HIGH_TIERS:
            priority_flags.append('tier_i_priority')
        if crl == 'High':
            priority_flags.append('high_crl_priority')
        records.append({
            'record_id': record_id,
            'title': first_nonempty(row, 'title', default='Untitled study'),
            'year': first_nonempty(row, 'publication_year', 'year', default='not_reported'),
            'tier': tier,
            'crl': crl,
            'study_design': first_nonempty(row, 'study_design', default='not_reported'),
            'wfs': first_nonempty(row, 'wfs', default='not_reported'),
            'wfs_primary': wfs_primary,
            'gi_subsite': first_nonempty(row, 'gi_subsite', default='general_gi'),
            'abstract': first_nonempty(row, 'abstract'),
            'priority_flags': '|'.join(priority_flags),
        })
    return records


def stratified_round_robin_sample(records: list[dict[str, Any]], target_n: int, seed: int) -> list[dict[str, Any]]:
    if target_n <= 0 or not records:
        return []
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record.get('tier', 'unknown'),
            record.get('crl', 'unknown'),
            record.get('gi_subsite', 'unknown'),
            record.get('wfs_primary', 'unknown'),
        )
        groups[key].append(record)
    ordered_keys = list(groups.keys())
    rng.shuffle(ordered_keys)
    for key in ordered_keys:
        rng.shuffle(groups[key])
    sampled: list[dict[str, Any]] = []
    while ordered_keys and len(sampled) < target_n:
        next_round: list[tuple[str, str, str, str]] = []
        for key in ordered_keys:
            bucket = groups[key]
            if bucket and len(sampled) < target_n:
                sampled.append(bucket.pop())
            if bucket:
                next_round.append(key)
        ordered_keys = next_round
    return sampled


def build_priority_subsets(
    records: list[dict[str, Any]],
    meta_ids_ordered: list[str],
    *,
    level2_max: int,
    level2_min: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    records_by_id = {record['record_id']: record for record in records}
    level1_records: list[dict[str, Any]] = []
    missing_level1_ids: list[str] = []
    for record_id in meta_ids_ordered:
        row = records_by_id.get(record_id)
        if row:
            enriched = dict(row)
            enriched['subset_level'] = 'level_1_meta'
            enriched['subset_reason'] = 'meta_input_truth_subset'
            enriched['priority_flags'] = '|'.join(sorted(set((row.get('priority_flags') or '').split('|') + ['meta_analysis_priority']) - {''}))
            level1_records.append(enriched)
        else:
            missing_level1_ids.append(record_id)

    level1_ids = {record['record_id'] for record in level1_records}
    level2_pool = [
        record for record in records
        if record['record_id'] not in level1_ids and (record['tier'] in LEVEL2_TIER_PRIORITY or record['crl'] == 'High')
    ]
    level2_target = min(max(level2_max, 0), len(level2_pool))
    if level2_target < min(level2_min, len(level2_pool)):
        level2_target = min(level2_min, len(level2_pool))
    sampled_level2 = stratified_round_robin_sample(level2_pool, level2_target, seed)
    level2_records: list[dict[str, Any]] = []
    for record in sampled_level2:
        enriched = dict(record)
        enriched['subset_level'] = 'level_2_stratified'
        enriched['subset_reason'] = 'stratified_tier_crl_gi_subsite_wfs'
        enriched['priority_flags'] = '|'.join(sorted(set((record.get('priority_flags') or '').split('|') + ['level2_priority']) - {''}))
        level2_records.append(enriched)
    return level1_records, level2_records, missing_level1_ids


def is_ai_study(record: dict[str, Any]) -> bool:
    haystack = ' '.join([
        str(record.get('title', '')),
        str(record.get('abstract', '')),
        str(record.get('study_design', '')),
        str(record.get('tier', '')),
    ]).lower()
    return any(token in haystack for token in AI_SIGNALS)


def assign_tool(record: dict[str, Any]) -> tuple[str, str]:
    design = str(record['study_design']).lower()
    text = f"{record['title']} {record['abstract']} {record['wfs']}".lower()
    if 'rct' in design or 'randomized' in text:
        return 'RoB 2', 'Study design explicitly reports randomized or controlled trial features.'
    if 'diagnostic' in design or any(token in text for token in DIAGNOSTIC_SIGNALS):
        return 'QUADAS-2', 'Study appears to evaluate diagnostic accuracy or test-performance endpoints.'
    if any(token in design for token in PREDICTION_SIGNALS) or any(token in text for token in PREDICTION_SIGNALS):
        if is_ai_study(record):
            return 'PROBAST+AI', 'Prediction/classification study uses AI/LLM methods, so PROBAST+AI is the best match.'
        return 'PROBAST', 'Prediction-model study without clear AI-specific features defaults to core PROBAST.'
    if any(token in design for token in OBSERVATIONAL_SIGNALS):
        return 'ROBINS-I', 'Non-randomized comparative or observational design best aligns with ROBINS-I.'
    return 'ROBINS-I', 'Defaulted to ROBINS-I because design is non-randomized or insufficiently specified.'


def worksheet_row(record: dict[str, Any], tool: str, reason: str) -> dict[str, Any]:
    row = {
        'record_id': record['record_id'],
        'title': record['title'],
        'year': record['year'],
        'tier': record['tier'],
        'crl': record['crl'],
        'gi_subsite': record['gi_subsite'],
        'wfs': record['wfs'],
        'wfs_primary': record['wfs_primary'],
        'study_design': record['study_design'],
        'priority_flags': record['priority_flags'],
        'subset_level': record['subset_level'],
        'subset_reason': record['subset_reason'],
        'assigned_tool': tool,
        'assignment_reason': reason,
        'reviewer_1': '',
        'reviewer_2': '',
        'adjudicator': '',
        'completion_status': 'pending',
        'overall_judgement': '',
        'rationale_summary': '',
        'notes': '',
    }
    for column in ALL_DOMAIN_COLUMNS:
        row[column] = 'NA'
    active_domains: list[str] = []
    if tool == 'RoB 2':
        active_domains = ROB2_DOMAINS
    elif tool == 'ROBINS-I':
        active_domains = ROBINS_I_DOMAINS
    elif tool == 'QUADAS-2':
        active_domains = QUADAS2_DOMAINS
    elif tool == 'PROBAST':
        active_domains = PROBAST_CORE_DOMAINS
    elif tool == 'PROBAST+AI':
        active_domains = PROBAST_DOMAINS
    for column in active_domains:
        row[column] = ''
    return row


def resolve_submission_contract_source(args: argparse.Namespace, output_dir: pathlib.Path) -> pathlib.Path:
    if args.formal_assessment_source:
        return pathlib.Path(args.formal_assessment_source)
    candidates = [
        output_dir / 'primary53_formal_assessment_submission_summary_2026-04-05.csv',
        output_dir / 'primary53_formal_assessment_workplan_2026-04-05_completed_workplan.csv',
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def normalize_submission_tool(row: dict[str, Any]) -> str:
    for key in ('adjudicated_final_status', 'final_status', 'tool_assigned'):
        value = first_nonempty(row, key)
        if value:
            return value
    return ''


def normalize_submission_result(row: dict[str, Any]) -> str:
    for key in ('adjudicated_overall_result_or_applicability', 'adjudicated_final', 'overall_judgement'):
        value = first_nonempty(row, key)
        if value:
            return value
    return ''


def build_submission_contract_outputs(
    *,
    input_records: list[dict[str, Any]],
    output_dir: pathlib.Path,
    source_path: pathlib.Path,
) -> dict[str, Any]:
    if not source_path.exists():
        raise SystemExit(f'--submission-contract requires a formal-assessment source CSV: {source_path}')

    with source_path.open('r', encoding='utf-8-sig', newline='') as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise SystemExit(f'formal-assessment source is empty: {source_path}')

    records_by_id = {record['record_id']: record for record in input_records}
    worksheet: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    missing_record_ids: list[str] = []

    for source_row in source_rows:
        record_id = normalize_record_id(first_nonempty(source_row, 'record_id'))
        if not record_id:
            continue
        record = records_by_id.get(record_id)
        if record is None:
            missing_record_ids.append(record_id)
            continue
        tool = normalize_submission_tool(source_row)
        overall = normalize_submission_result(source_row)
        if not tool or not overall:
            raise SystemExit(f'submission-contract source missing tool/result for record {record_id}')
        reason = first_nonempty(source_row, 'formal_non_applicable_reason', default='')
        submission_row = worksheet_row(
            {
                **record,
                'subset_level': 'submission_contract',
                'subset_reason': 'adjudicated_formal_assessment_authority',
                'priority_flags': 'submission_contract',
            },
            tool,
            reason or 'submission-facing formal-assessment authority',
        )
        submission_row['completion_status'] = 'completed'
        submission_row['overall_judgement'] = overall
        submission_row['rationale_summary'] = reason or 'n/a - formal tool completed'
        worksheet.append(submission_row)
        assignments.append({
            'record_id': record_id,
            'subset_level': 'submission_contract',
            'subset_reason': 'adjudicated_formal_assessment_authority',
            'assigned_tool': tool,
            'reason': reason or 'submission-facing formal-assessment authority',
            'priority_flags': 'submission_contract',
        })
        subset_rows.append({
            'record_id': record_id,
            'title': record['title'],
            'year': record['year'],
            'tier': record['tier'],
            'crl': record['crl'],
            'gi_subsite': record['gi_subsite'],
            'wfs': record['wfs'],
            'wfs_primary': record['wfs_primary'],
            'study_design': record['study_design'],
            'subset_level': 'submission_contract',
            'subset_reason': 'adjudicated_formal_assessment_authority',
            'priority_flags': 'submission_contract',
            'assigned_tool': tool,
        })

    worksheet_path = output_dir / 'rob_assessment_worksheet.csv'
    assignment_path = output_dir / 'rob_tool_assignment.json'
    combined_subset_path = output_dir / 'rob_priority_subset_combined.csv'
    legacy_subset_path = output_dir / 'rob_priority_subset.csv'
    level1_subset_path = output_dir / 'rob_priority_subset_level1.csv'
    level2_subset_path = output_dir / 'rob_priority_subset_level2.csv'
    completion_path = output_dir / 'rob_completion_status.json'

    subset_fields = [
        'record_id',
        'title',
        'year',
        'tier',
        'crl',
        'gi_subsite',
        'wfs',
        'wfs_primary',
        'study_design',
        'subset_level',
        'subset_reason',
        'priority_flags',
        'assigned_tool',
    ]
    worksheet_fields = [
        'record_id',
        'title',
        'year',
        'tier',
        'crl',
        'gi_subsite',
        'wfs',
        'wfs_primary',
        'study_design',
        'priority_flags',
        'subset_level',
        'subset_reason',
        'assigned_tool',
        'assignment_reason',
        'reviewer_1',
        'reviewer_2',
        'adjudicator',
        'completion_status',
        'overall_judgement',
        'rationale_summary',
        'notes',
    ] + ALL_DOMAIN_COLUMNS
    write_csv(worksheet_path, worksheet_fields, worksheet)
    write_csv(level1_subset_path, subset_fields, subset_rows)
    write_csv(level2_subset_path, subset_fields, [])
    write_csv(combined_subset_path, subset_fields, subset_rows)
    write_csv(legacy_subset_path, subset_fields, subset_rows)
    assignment_path.write_text(json.dumps(assignments, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    tool_counts = Counter(item['assigned_tool'] for item in assignments)
    summary = {
        'status': 'submission_contract_from_adjudicated_authority',
        'source_path': str(source_path),
        'submission_contract_record_n': len(subset_rows),
        'tool_assignment_counts': dict(sorted(tool_counts.items())),
        'missing_record_ids_from_input': sorted(set(missing_record_ids)),
        'paths': {
            'worksheet_path': str(worksheet_path),
            'rob_tool_assignment': str(assignment_path),
            'level1_subset_path': str(level1_subset_path),
            'level2_subset_path': str(level2_subset_path),
            'combined_subset_path': str(combined_subset_path),
            'legacy_subset_path': str(legacy_subset_path),
        },
        'notes': [
            'Submission-contract mode mirrors the adjudicated formal-assessment authority only.',
            'Legacy heuristic tool assignment families (RoB 2, ROBINS-I, QUADAS-2) are not used in the submission-clean path.',
        ],
    }
    completion_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return summary


def main() -> None:
    args = parse_args()
    if not args.demo and not args.input:
        raise SystemExit('--input is required unless --demo is used')
    if args.level2_max < 0:
        raise SystemExit('--level2-max must be >= 0')
    if args.level2_min < 0:
        raise SystemExit('--level2-min must be >= 0')

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        raw_rows = build_demo_rows()
    else:
        with open(args.input, 'r', encoding='utf-8-sig', newline='') as handle:
            raw_rows = list(csv.DictReader(handle))

    records = normalize_records(raw_rows)
    if not records:
        raise SystemExit('No valid tier/crl records available for RoB subset generation')
    if args.submission_contract:
        summary = build_submission_contract_outputs(
            input_records=records,
            output_dir=output_dir,
            source_path=resolve_submission_contract_source(args, output_dir),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    meta_ids_ordered, meta_source_path = load_meta_ids(args.meta_input, output_dir)
    if args.demo and not meta_ids_ordered:
        meta_ids_ordered = [normalize_record_id(raw_rows[index]['record_id']) for index in range(min(39, len(raw_rows)))]

    level1_records, level2_records, missing_level1_ids = build_priority_subsets(
        records,
        meta_ids_ordered,
        level2_max=args.level2_max,
        level2_min=args.level2_min,
        seed=args.seed,
    )
    combined_records = level1_records + level2_records

    worksheet = []
    assignments = []
    for record in combined_records:
        tool, reason = assign_tool(record)
        worksheet.append(worksheet_row(record, tool, reason))
        assignments.append({
            'record_id': record['record_id'],
            'subset_level': record['subset_level'],
            'subset_reason': record['subset_reason'],
            'assigned_tool': tool,
            'reason': reason,
            'priority_flags': record['priority_flags'],
        })

    def subset_rows(rows: list[dict[str, Any]], assignments_by_id: dict[str, str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append({
                'record_id': row['record_id'],
                'title': row['title'],
                'year': row['year'],
                'tier': row['tier'],
                'crl': row['crl'],
                'gi_subsite': row['gi_subsite'],
                'wfs': row['wfs'],
                'wfs_primary': row['wfs_primary'],
                'study_design': row['study_design'],
                'subset_level': row['subset_level'],
                'subset_reason': row['subset_reason'],
                'priority_flags': row['priority_flags'],
                'assigned_tool': assignments_by_id.get(row['record_id'], ''),
            })
        return out

    assignments_by_id = {item['record_id']: item['assigned_tool'] for item in assignments}
    level1_rows = subset_rows(level1_records, assignments_by_id)
    level2_rows = subset_rows(level2_records, assignments_by_id)
    combined_rows = subset_rows(combined_records, assignments_by_id)

    worksheet_path = output_dir / 'rob_assessment_worksheet.csv'
    assignment_path = output_dir / 'rob_tool_assignment.json'
    combined_subset_path = output_dir / 'rob_priority_subset_combined.csv'
    legacy_subset_path = output_dir / 'rob_priority_subset.csv'
    level1_subset_path = output_dir / 'rob_priority_subset_level1.csv'
    level2_subset_path = output_dir / 'rob_priority_subset_level2.csv'
    completion_path = output_dir / 'rob_completion_status.json'

    subset_fields = [
        'record_id',
        'title',
        'year',
        'tier',
        'crl',
        'gi_subsite',
        'wfs',
        'wfs_primary',
        'study_design',
        'subset_level',
        'subset_reason',
        'priority_flags',
        'assigned_tool',
    ]
    worksheet_fields = [
        'record_id',
        'title',
        'year',
        'tier',
        'crl',
        'gi_subsite',
        'wfs',
        'wfs_primary',
        'study_design',
        'priority_flags',
        'subset_level',
        'subset_reason',
        'assigned_tool',
        'assignment_reason',
        'reviewer_1',
        'reviewer_2',
        'adjudicator',
        'completion_status',
        'overall_judgement',
        'rationale_summary',
        'notes',
    ] + ALL_DOMAIN_COLUMNS
    write_csv(worksheet_path, worksheet_fields, worksheet)
    write_csv(level1_subset_path, subset_fields, level1_rows)
    write_csv(level2_subset_path, subset_fields, level2_rows)
    write_csv(combined_subset_path, subset_fields, combined_rows)
    write_csv(legacy_subset_path, subset_fields, combined_rows)
    assignment_path.write_text(json.dumps(assignments, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    tool_counts = Counter(item['assigned_tool'] for item in assignments)
    summary = {
        'total_records_available': len(records),
        'level_1_meta_target_n': len(meta_ids_ordered),
        'level_1_meta_matched_n': len(level1_rows),
        'level_1_meta_missing_record_ids': missing_level1_ids,
        'level_2_candidate_pool_n': len([record for record in records if record['record_id'] not in {row['record_id'] for row in level1_records} and (record['tier'] in LEVEL2_TIER_PRIORITY or record['crl'] == 'High')]),
        'level_2_requested_min': args.level2_min,
        'level_2_requested_max': args.level2_max,
        'level_2_selected_n': len(level2_rows),
        'combined_subset_n': len(combined_rows),
        'subset_level_counts': {
            'level_1_meta': len(level1_rows),
            'level_2_stratified': len(level2_rows),
        },
        'tool_assignment_counts': {
            'RoB 2': tool_counts.get('RoB 2', 0),
            'ROBINS-I': tool_counts.get('ROBINS-I', 0),
            'QUADAS-2': tool_counts.get('QUADAS-2', 0),
            'PROBAST': tool_counts.get('PROBAST', 0),
            'PROBAST+AI': tool_counts.get('PROBAST+AI', 0),
        },
        'meta_input_source': str(meta_source_path) if meta_source_path else '',
        'status': 'ready_for_completion_worksheet',
        'completed_domain_level_judgments_n': 0,
        'pending_manual_judgment_n': len(combined_rows),
        'paths': {
            'worksheet_path': str(worksheet_path),
            'rob_tool_assignment': str(assignment_path),
            'level1_subset_path': str(level1_subset_path),
            'level2_subset_path': str(level2_subset_path),
            'combined_subset_path': str(combined_subset_path),
            'legacy_subset_path': str(legacy_subset_path),
        },
        'notes': [
            'This output prepares adjudication worksheets and does not claim completed domain-level RoB judgments.',
            'Level-1 strictly follows meta_input truth by record_id matching.',
            'Level-2 is stratified expansion over Tier I-a/I-b and High CRL.',
        ],
    }
    completion_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
