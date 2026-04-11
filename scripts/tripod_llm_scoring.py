from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from typing import Any

from llm_screening import (
    MAX_RETRIES,
    RATE_LIMIT_SECONDS,
    ApiError,
    anthropic_rest_call,
    anthropic_sdk_call,
    extract_json_object,
    load_csv,
    now_utc,
    openai_rest_call,
    openai_sdk_call,
    save_response,
    write_csv,
)
from pipeline_lib import ANTHROPIC_MODEL, OPENAI_MODEL, ensure_columns as ensure_extra_columns

TRIPOD_ITEMS = [
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
PROMPT_TEMPLATE = '''You are a reporting-quality assessor for TRIPOD-LLM compliance.
Assess whether each checklist item is explicitly reported in the study title or abstract.

Study title: {title}
Study abstract: {abstract}
Tier: {tier}

Return JSON in this exact schema:
{{
  "title_identifies_llm": 0,
  "abstract_structured": 0,
  "objectives_stated": 0,
  "data_source_described": 0,
  "participants_described": 0,
  "outcome_defined": 0,
  "llm_selection_justified": 0,
  "llm_version_reported": 0,
  "prompt_engineering_described": 0,
  "llm_config_reported": 0,
  "sample_size_justified": 0,
  "missing_data_addressed": 0,
  "statistical_methods_described": 0,
  "risk_of_bias_assessed": 0,
  "participant_flow_reported": 0,
  "performance_metrics_reported": 0,
  "model_updating_discussed": 0,
  "limitations_discussed": 0,
  "clinical_implications_stated": 0,
  "brief_rationale": "1-3 sentences"
}}
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Score TRIPOD-LLM reporting compliance from title/abstract text.')
    parser.add_argument('--input', help='Input CSV path with labeled studies')
    parser.add_argument('--output-dir', required=True, help='Directory for TRIPOD outputs')
    parser.add_argument('--provider', default='openai', choices=['openai', 'anthropic'])
    parser.add_argument('--run-id', default='tripod-demo-run')
    parser.add_argument('--only-included', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--demo', action='store_true')
    return parser.parse_args()


def demo_rows() -> tuple[list[str], list[dict[str, str]]]:
    fieldnames = ['record_id', 'title', 'abstract', 'tier', 'decision']
    rows = [
        {
            'record_id': 'TRIPOD001',
            'title': 'GPT-4 for colorectal cancer MDT planning: a prospective comparative study',
            'abstract': 'Background: We evaluated GPT-4-turbo-2024-04-09 for colorectal MDT planning. Methods: In 120 patients we compared GPT-4 recommendations with clinicians using accuracy, F1 score, and chi-square tests. Results: Accuracy was 0.84. Limitations and clinical implications are discussed.',
            'tier': 'I-a',
            'decision': 'Include',
        },
        {
            'record_id': 'TRIPOD002',
            'title': 'LLM benchmark for gastric oncology questions',
            'abstract': 'We tested ChatGPT on exam-style gastric oncology questions and report accuracy.',
            'tier': 'III',
            'decision': 'Include',
        },
    ]
    return fieldnames, rows


def heuristic_score(row: dict[str, str]) -> dict[str, Any]:
    title = (row.get('title') or '').lower()
    abstract = (row.get('abstract') or '').lower()
    text = f'{title} {abstract}'
    values = {
        'title_identifies_llm': int(any(token in title for token in ['gpt', 'chatgpt', 'claude', 'gemini', 'llm', 'ai'])),
        'abstract_structured': int(any(token in abstract for token in ['background:', 'methods:', 'results:', 'conclusion'])),
        'objectives_stated': int(any(token in text for token in ['aim', 'objective', 'evaluate', 'assess'])),
        'data_source_described': int(any(token in text for token in ['dataset', 'patients', 'cases', 'registry'])),
        'participants_described': int(any(token in text for token in ['patient', 'participants', 'cases'])),
        'outcome_defined': int(any(token in text for token in ['accuracy', 'f1', 'auc', 'sensitivity', 'specificity'])),
        'llm_selection_justified': int(any(token in text for token in ['selected', 'chosen', 'because', 'reason'])),
        'llm_version_reported': int(any(token in text for token in ['gpt-4-', '0613', '2024-', 'turbo'])),
        'prompt_engineering_described': int('prompt' in text),
        'llm_config_reported': int(any(token in text for token in ['temperature', 'top_p', 'parameter'])),
        'sample_size_justified': int(any(token in text for token in ['sample size', 'powered', 'power analysis'])),
        'missing_data_addressed': int('missing data' in text),
        'statistical_methods_described': int(any(token in text for token in ['chi-square', 'logistic regression', 'mann-whitney', 'statistical'])),
        'risk_of_bias_assessed': int(any(token in text for token in ['risk of bias', 'rob', 'probast', 'quadas'])),
        'participant_flow_reported': int(any(token in text for token in ['flow', 'consort', 'screened', 'included'])),
        'performance_metrics_reported': int(any(token in text for token in ['accuracy', 'auc', 'f1', 'precision', 'recall'])),
        'model_updating_discussed': int(any(token in text for token in ['updated', 'version change', 'model drift'])),
        'limitations_discussed': int('limitation' in text),
        'clinical_implications_stated': int(any(token in text for token in ['clinical', 'practice', 'patient care', 'implication'])),
    }
    values['brief_rationale'] = 'Heuristic demo scoring from title/abstract keywords.'
    return values


def normalize_result(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for item in TRIPOD_ITEMS:
        value = payload.get(item, 0)
        if value not in {0, 1}:
            raise ApiError(f'Invalid TRIPOD item value for {item}: {value!r}')
        normalized[item] = int(value)
    normalized['brief_rationale'] = str(payload.get('brief_rationale', '')).strip() or 'No rationale provided.'
    return normalized


def call_provider(provider: str, prompt: str, record_context: str = '') -> tuple[dict[str, Any], dict[str, Any]]:
    context_note = f' record={record_context}' if record_context else ''
    if provider == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ApiError('OPENAI_API_KEY is not set')
        sdk_error: Exception | None = None
        try:
            raw, text = openai_sdk_call(prompt, api_key)
            method = 'sdk'
        except Exception as exc:
            sdk_error = exc
            try:
                raw, text = openai_rest_call(prompt, api_key)
                method = 'rest'
            except Exception as rest_error:
                raise ApiError(f'SDK error: {sdk_error}; REST error: {rest_error}{context_note}') from rest_error
        try:
            parsed = normalize_result(extract_json_object(text))
        except Exception as exc:
            raise ApiError(f'JSON parse/validation error{context_note}: {exc}') from exc
        return parsed, {'provider': provider, 'model': OPENAI_MODEL, 'transport': method, 'response': raw, 'response_text': text}
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ApiError('ANTHROPIC_API_KEY is not set')
    sdk_error = None
    try:
        raw, text = anthropic_sdk_call(prompt, api_key)
        method = 'sdk'
    except Exception as exc:
        sdk_error = exc
        try:
            raw, text = anthropic_rest_call(prompt, api_key)
            method = 'rest'
        except Exception as rest_error:
            raise ApiError(f'SDK error: {sdk_error}; REST error: {rest_error}{context_note}') from rest_error
    try:
        parsed = normalize_result(extract_json_object(text))
    except Exception as exc:
        raise ApiError(f'JSON parse/validation error{context_note}: {exc}') from exc
    return parsed, {'provider': provider, 'model': ANTHROPIC_MODEL, 'transport': method, 'response': raw, 'response_text': text}


def should_skip(row: dict[str, str], resume: bool, only_included: bool) -> bool:
    if only_included and (row.get('decision') or '').strip() != 'Include':
        return True
    if not resume:
        return False
    return bool((row.get('total_score') or '').strip())


def build_prompt(row: dict[str, str]) -> str:
    return PROMPT_TEMPLATE.format(
        title=(row.get('title') or '').strip(),
        abstract=(row.get('abstract') or '').strip(),
        tier=(row.get('tier') or '').strip(),
    )


def failure_payload(message: str) -> dict[str, Any]:
    payload = {item: 'error' for item in TRIPOD_ITEMS}
    payload['brief_rationale'] = message
    return payload


def compute_scores(result: dict[str, Any]) -> tuple[int | str, float | str]:
    raw_values = [result.get(item, 'error') for item in TRIPOD_ITEMS]
    if any(str(value).strip().lower() == 'error' for value in raw_values):
        return 'error', 'error'
    values = [int(value) for value in raw_values]
    total_score = sum(values)
    return total_score, round(total_score / len(TRIPOD_ITEMS) * 100.0, 2)


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    valid_totals = []
    for row in rows:
        total_value = str(row.get('total_score', '')).strip()
        if not total_value or total_value == 'error':
            continue
        valid_totals.append(int(total_value))
    reporting_rates = {}
    for item in TRIPOD_ITEMS:
        valid_values = []
        for row in rows:
            value = str(row.get(item, '')).strip()
            if not value or value == 'error':
                continue
            valid_values.append(int(value))
        reporting_rates[item] = round((sum(valid_values) / len(valid_values) * 100.0) if valid_values else 0.0, 2)
    return {
        'study_count': len(rows),
        'valid_scored_studies': len(valid_totals),
        'mean_total_score': round(statistics.mean(valid_totals), 4) if valid_totals else 0.0,
        'median_total_score': round(statistics.median(valid_totals), 4) if valid_totals else 0.0,
        'sd_total_score': round(statistics.pstdev(valid_totals), 4) if len(valid_totals) > 1 else 0.0,
        'reporting_rate_percent': reporting_rates,
    }


def main() -> None:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_dir = pathlib.Path(__file__).resolve().parent
    project_dir = script_dir.parent
    response_dir = project_dir / 'logs' / 'runs' / args.run_id / 'tripod' / 'llm_responses'

    if args.demo:
        fieldnames, rows = demo_rows()
    else:
        if not args.input:
            raise SystemExit('--input is required unless --demo is used')
        fieldnames, rows = load_csv(pathlib.Path(args.input))
    fieldnames = ensure_extra_columns(fieldnames, rows, TRIPOD_ITEMS + ['total_score', 'compliance_rate'])

    indices = [index for index, row in enumerate(rows) if not should_skip(row, args.resume, args.only_included)]
    last_request_time = 0.0
    total = len(indices)

    for processed_count, row_index in enumerate(indices, start=1):
        row = rows[row_index]
        record_id = (row.get('record_id') or f'row_{row_index + 1}').strip() or f'row_{row_index + 1}'
        prompt = build_prompt(row)
        attempt = 0
        while True:
            attempt += 1
            elapsed = time.monotonic() - last_request_time
            if elapsed < RATE_LIMIT_SECONDS:
                time.sleep(RATE_LIMIT_SECONDS - elapsed)
            try:
                if args.demo:
                    result = heuristic_score(row)
                    raw_payload = {'provider': 'demo', 'model': 'heuristic', 'transport': 'demo', 'response': {'demo': True}, 'response_text': json.dumps(result, ensure_ascii=False)}
                else:
                    result, raw_payload = call_provider(args.provider, prompt, record_id)
                last_request_time = time.monotonic()
                break
            except Exception as exc:
                last_request_time = time.monotonic()
                if attempt >= MAX_RETRIES:
                    print(f'[tripod_llm_scoring] WARNING: {record_id} failed after {MAX_RETRIES} attempts: {exc}', file=sys.stderr, flush=True)
                    result = failure_payload(f'API error after {MAX_RETRIES} retries: {exc}')
                    raw_payload = {'provider': args.provider, 'model': 'error', 'transport': 'error', 'response': {'error': str(exc)}, 'response_text': ''}
                    break
                time.sleep(2 ** (attempt - 1))

        total_score, compliance_rate = compute_scores(result)
        for item in TRIPOD_ITEMS:
            row[item] = str(result[item])
        row['total_score'] = str(total_score)
        row['compliance_rate'] = str(compliance_rate)

        save_response(
            response_dir / f'{record_id}.json',
            {
                'processed_at': now_utc(),
                'record_id': record_id,
                'provider': raw_payload['provider'],
                'model': raw_payload['model'],
                'transport': raw_payload['transport'],
                'prompt': prompt,
                'parsed_result': result,
                'raw_response': raw_payload['response'],
                'response_text': raw_payload['response_text'],
            },
        )
        write_csv(output_dir / 'tripod_scores.csv', fieldnames, rows)
        if processed_count % 50 == 0 or processed_count == total:
            print(f'[tripod_llm_scoring] processed {processed_count}/{total}', flush=True)

    summary = summarize(rows)
    summary_path = output_dir / 'tripod_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'[tripod_llm_scoring] scores={output_dir / "tripod_scores.csv"}')
    print(f'[tripod_llm_scoring] summary={summary_path}')


if __name__ == '__main__':
    main()
