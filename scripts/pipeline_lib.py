from __future__ import annotations

import csv
import json
import pathlib
from datetime import datetime, timezone

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
RUNS_DIR = PROJECT_DIR / 'logs' / 'runs'
RAW_DIR = PROJECT_DIR / 'data' / 'raw'
INTERIM_DIR = PROJECT_DIR / 'data' / 'interim'
PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
EXPORTS_DIR = PROJECT_DIR / 'exports'
OUTPUTS_DIR = PROJECT_DIR / 'outputs'
ORCHESTRATION_DIR = PROJECT_DIR / 'orchestration'
STUDY_RECORD_TEMPLATE = PROJECT_DIR / 'schemas' / 'study_record_template.csv'

PEER_REVIEWED_SOURCES = {
    'pubmed',
    'medline',
    'scopus',
}

B_ONLY_SOURCES = {
    'medrxiv',
    'biorxiv',
    'arxiv',
    'clinicaltrials.gov',
    'who ictrp',
    'wh oictrp',
    'chictr',
    'google scholar',
    'googlescholar',
    'citation chaining',
    'citationchaining',
    'cnki',
}

SCREENING_COLUMNS = [
    'record_id',
    'protocol',
    'source_database',
    'source_database_detail',
    'import_batch_id',
    'title',
    'abstract',
    'publication_year',
    'doi',
    'journal_or_source',
    'decision',
    'rationale_code',
    'rationale_text',
    'confidence',
    'needs_human_review',
    'title_abstract_decision',
    'screening_rationale_short',
    'reviewer_id',
    'adjudication_needed',
    'full_text_decision',
    'exclusion_codes',
    'notes',
]

SOURCE_MANIFEST_REQUIRED = [
    'run_id',
    'protocol',
    'source_database',
    'search_date',
    'query_version',
    'native_export_path',
    'normalized_csv_path',
    'record_count_raw',
    'record_count_imported',
    'status',
]

STAGE_ORDER = [
    'freeze',
    'protocol_a_data',
    'protocol_a_screening',
    'protocol_a_extraction_analysis',
    'protocol_b_data',
    'protocol_b_screening',
    'protocol_b_extraction_analysis',
    'ab_comparison_bundle',
]


def load_template_fields() -> list[str]:
    with open(STUDY_RECORD_TEMPLATE, 'r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        return next(reader)


STUDY_RECORD_FIELDS = load_template_fields()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def ensure_parent(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv(path: pathlib.Path | str) -> list[dict[str, str]]:
    with open(path, 'r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: pathlib.Path | str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path = pathlib.Path(path)
    ensure_parent(path)
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: pathlib.Path | str) -> dict:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path: pathlib.Path | str, data: dict) -> None:
    path = pathlib.Path(path)
    ensure_parent(path)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def csv_row_count(path: pathlib.Path | str) -> int:
    with open(path, 'r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def validate_required_keys(data: dict, required: list[str], label: str) -> None:
    missing = [key for key in required if key not in data or data[key] in (None, '')]
    if missing:
        raise ValueError(f'{label} missing required keys: {", ".join(missing)}')


def repo_relative(path: pathlib.Path | str) -> str:
    resolved = pathlib.Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def resolve_path(value: str | pathlib.Path, *, base_dir: pathlib.Path | None = None) -> pathlib.Path:
    raw_value = str(value)
    if '\\' in raw_value:
        raw_value = raw_value.replace('\\', '/')
    path = pathlib.Path(raw_value)
    if path.is_absolute():
        return path
    if base_dir is not None:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return candidate
    return (REPO_ROOT / path).resolve()


def normalize_text(text: str | None) -> str:
    return ' '.join((text or '').strip().split())


def infer_protocol(source_name: str) -> str:
    normalized = source_name.strip().lower()
    if normalized in PEER_REVIEWED_SOURCES:
        return 'A'
    if normalized in B_ONLY_SOURCES:
        return 'B'
    return 'B'


def peer_review_status(protocol: str) -> str:
    return 'peer_reviewed' if protocol == 'A' else 'mixed_or_non_peer_reviewed'


def normalize_export_row(row: dict[str, str]) -> dict[str, str]:
    return {str(key).strip().lower(): normalize_text(value) for key, value in row.items()}


def first_present(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = normalize_text(row.get(key))
        if value:
            return value
    return ''


def build_source_provenance(manifest: dict) -> str:
    source = manifest.get('source_database', '')
    detail = manifest.get('source_database_detail') or manifest.get('query_version', '')
    run_id = manifest.get('run_id', '')
    parts = [part for part in [source, detail, run_id] if part]
    return ' | '.join(parts)


def record_from_source_row(row: dict[str, str], *, source_name: str, record_index: int, manifest: dict) -> dict[str, str]:
    normalized = normalize_export_row(row)
    protocol = manifest.get('protocol') or infer_protocol(source_name)
    output = {field: '' for field in STUDY_RECORD_FIELDS}
    output.update({
        'record_id': f'R{record_index:06d}',
        'protocol': protocol,
        'source_database': source_name,
        'source_database_detail': manifest.get('source_database_detail') or manifest.get('query_version', ''),
        'import_batch_id': manifest.get('run_id', ''),
        'title': first_present(normalized, ['title', 'article title', 'document title']),
        'abstract': first_present(normalized, ['abstract', 'abstract note', 'summary']),
        'authors': first_present(normalized, ['authors', 'author full names', 'author']),
        'journal_or_source': first_present(normalized, ['journal', 'journal/source title', 'source title', 'source']),
        'publication_year': first_present(normalized, ['year', 'publication year', 'py']),
        'doi': first_present(normalized, ['doi', 'document object identifier (doi)']),
        'pmid': first_present(normalized, ['pmid']),
        'other_id': first_present(normalized, ['eid', 'accession number', 'ut', 'trial id']),
        'language': first_present(normalized, ['language']) or 'unknown',
        'publication_type': first_present(normalized, ['publication type', 'document type']),
        'peer_review_status': manifest.get('peer_review_status') or peer_review_status(protocol),
        'full_text_available': manifest.get('full_text_available', ''),
        'url': first_present(normalized, ['url', 'source url', 'link']),
        'protocol_a_eligible': 'pending',
        'protocol_b_eligible': 'pending',
        'notes': build_source_provenance(manifest),
    })
    return output


def build_stage_state(
    *,
    stage_name: str,
    run_id: str,
    protocol: str | None,
    status: str,
    input_dependencies: list[str],
    pass_criteria: list[str],
    failure_reasons: list[str],
    human_gate_status: str,
    artifact_paths: dict,
    notes: list[str] | None = None,
) -> dict:
    return {
        'stage_name': stage_name,
        'run_id': run_id,
        'protocol': protocol,
        'status': status,
        'updated_at': now_utc(),
        'input_dependencies': input_dependencies,
        'pass_criteria': pass_criteria,
        'failure_reasons': failure_reasons,
        'human_gate_status': human_gate_status,
        'artifact_paths': artifact_paths,
        'notes': notes or [],
    }

# === Evidence Framework Constants ===

TIERS = ['S', 'I-a', 'I-b', 'II', 'III']

CRL_LEVELS = ['Low', 'Medium', 'High']

WFS_ORDER = [
    'screening',
    'diagnosis',
    'staging',
    'mdt',
    'treatment',
    'perioperative',
    'followup',
    'patient_communication',
    'research_qc',
]

GI_SUBSITE_ORDER = [
    'esophageal',
    'gastric',
    'colorectal',
    'small_bowel',
    'anal',
    'multiple_gi',
    'general_gi',
]

OPENAI_MODEL = 'gpt-5.4'
ANTHROPIC_MODEL = 'claude-sonnet-4-6'


def ensure_columns(fieldnames: list[str], rows: list[dict[str, str]], extra_columns: list[str]) -> list[str]:
    final_fields = list(fieldnames)
    for column in extra_columns:
        if column not in final_fields:
            final_fields.append(column)
    for row in rows:
        for column in final_fields:
            row.setdefault(column, '')
    return final_fields

