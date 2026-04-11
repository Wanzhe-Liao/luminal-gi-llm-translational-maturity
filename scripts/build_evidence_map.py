from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import struct
import textwrap
import zlib
from collections import Counter
from typing import Any

from pipeline_lib import CRL_LEVELS, GI_SUBSITE_ORDER, TIERS, WFS_ORDER

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

TIER_COLORS_HEX = {
    'S': '#0B6E4F',
    'I-a': '#1F8A70',
    'I-b': '#4C9F70',
    'II': '#D98E04',
    'III': '#A23E48',
}
CRL_COLORS_HEX = {
    'Low': '#6BAED6',
    'Medium': '#F4A261',
    'High': '#D1495B',
}
WFS_ABBR = {
    'screening': 'SCR',
    'diagnosis': 'DX',
    'staging': 'STG',
    'mdt': 'MDT',
    'treatment': 'TX',
    'perioperative': 'PERI',
    'followup': 'FUP',
    'patient_communication': 'COMM',
    'research_qc': 'QC',
}
SUBSITE_ABBR = {
    'esophageal': 'ESO',
    'gastric': 'GAS',
    'colorectal': 'CRC',
    'small_bowel': 'SBO',
    'anal': 'ANAL',
    'multiple_gi': 'MULTI',
    'general_gi': 'GI',
}
FONT = {
    ' ': ['00000', '00000', '00000', '00000', '00000', '00000', '00000'],
    '-': ['00000', '00000', '00000', '11111', '00000', '00000', '00000'],
    '0': ['01110', '10001', '10011', '10101', '11001', '10001', '01110'],
    '1': ['00100', '01100', '00100', '00100', '00100', '00100', '01110'],
    '2': ['01110', '10001', '00001', '00010', '00100', '01000', '11111'],
    '3': ['11110', '00001', '00001', '01110', '00001', '00001', '11110'],
    '4': ['00010', '00110', '01010', '10010', '11111', '00010', '00010'],
    '5': ['11111', '10000', '10000', '11110', '00001', '00001', '11110'],
    '6': ['00110', '01000', '10000', '11110', '10001', '10001', '01110'],
    '7': ['11111', '00001', '00010', '00100', '01000', '01000', '01000'],
    '8': ['01110', '10001', '10001', '01110', '10001', '10001', '01110'],
    '9': ['01110', '10001', '10001', '01111', '00001', '00010', '11100'],
    'A': ['01110', '10001', '10001', '11111', '10001', '10001', '10001'],
    'B': ['11110', '10001', '10001', '11110', '10001', '10001', '11110'],
    'C': ['01111', '10000', '10000', '10000', '10000', '10000', '01111'],
    'D': ['11110', '10001', '10001', '10001', '10001', '10001', '11110'],
    'E': ['11111', '10000', '10000', '11110', '10000', '10000', '11111'],
    'F': ['11111', '10000', '10000', '11110', '10000', '10000', '10000'],
    'G': ['01111', '10000', '10000', '10011', '10001', '10001', '01110'],
    'H': ['10001', '10001', '10001', '11111', '10001', '10001', '10001'],
    'I': ['11111', '00100', '00100', '00100', '00100', '00100', '11111'],
    'J': ['00001', '00001', '00001', '00001', '10001', '10001', '01110'],
    'K': ['10001', '10010', '10100', '11000', '10100', '10010', '10001'],
    'L': ['10000', '10000', '10000', '10000', '10000', '10000', '11111'],
    'M': ['10001', '11011', '10101', '10101', '10001', '10001', '10001'],
    'N': ['10001', '11001', '10101', '10011', '10001', '10001', '10001'],
    'O': ['01110', '10001', '10001', '10001', '10001', '10001', '01110'],
    'P': ['11110', '10001', '10001', '11110', '10000', '10000', '10000'],
    'Q': ['01110', '10001', '10001', '10001', '10101', '10010', '01101'],
    'R': ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
    'S': ['01111', '10000', '10000', '01110', '00001', '00001', '11110'],
    'T': ['11111', '00100', '00100', '00100', '00100', '00100', '00100'],
    'U': ['10001', '10001', '10001', '10001', '10001', '10001', '01110'],
    'V': ['10001', '10001', '10001', '10001', '10001', '01010', '00100'],
    'W': ['10001', '10001', '10001', '10101', '10101', '10101', '01010'],
    'X': ['10001', '10001', '01010', '00100', '01010', '10001', '10001'],
    'Y': ['10001', '10001', '01010', '00100', '00100', '00100', '00100'],
    'Z': ['11111', '00001', '00010', '00100', '01000', '10000', '11111'],
}


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip('#')
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


TIER_COLORS = {key: hex_to_rgb(value) for key, value in TIER_COLORS_HEX.items()}
CRL_COLORS = {key: hex_to_rgb(value) for key, value in CRL_COLORS_HEX.items()}
AXIS_COLOR = (45, 45, 45)
GRID_COLOR = (220, 223, 227)
TEXT_COLOR = (25, 25, 25)
LIGHT_TEXT_COLOR = (245, 245, 245)
BACKGROUND = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build evidence-map figures and summary tables from LLM tier labels.')
    parser.add_argument('--input', help='Input labeled CSV path')
    parser.add_argument('--output-dir', required=True, help='Directory for figures and tables')
    parser.add_argument('--title', default='LLMs in GI Oncology: Translational Evidence Map', help='Figure title prefix')
    parser.add_argument('--demo', action='store_true', help='Generate simulated labeled data and run the full pipeline')
    return parser.parse_args()


class SimpleCanvas:
    def __init__(self, width: int, height: int, bg: tuple[int, int, int] = BACKGROUND) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(width * height * 3)
        self.fill_rect(0, 0, width, height, bg)

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = (y * self.width + x) * 3
            self.pixels[idx:idx + 3] = bytes(color)

    def fill_rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + w)
        y1 = min(self.height, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        span = bytes(color) * (x1 - x0)
        for yy in range(y0, y1):
            row_start = (yy * self.width + x0) * 3
            row_end = row_start + (x1 - x0) * 3
            self.pixels[row_start:row_end] = span

    def draw_line(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def draw_rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        self.draw_line(x, y, x + w, y, color)
        self.draw_line(x, y, x, y + h, color)
        self.draw_line(x + w, y, x + w, y + h, color)
        self.draw_line(x, y + h, x + w, y + h, color)

    def draw_text(self, x: int, y: int, text: str, color: tuple[int, int, int] = TEXT_COLOR, scale: int = 2) -> None:
        cursor = x
        for char in sanitize_text(text):
            pattern = FONT.get(char, FONT[' '])
            for row_idx, row in enumerate(pattern):
                for col_idx, bit in enumerate(row):
                    if bit == '1':
                        self.fill_rect(cursor + col_idx * scale, y + row_idx * scale, scale, scale, color)
            cursor += (len(pattern[0]) + 1) * scale

    def save_png(self, path: pathlib.Path) -> None:
        raw = bytearray()
        stride = self.width * 3
        for row in range(self.height):
            raw.append(0)
            start = row * stride
            raw.extend(self.pixels[start:start + stride])
        compressed = zlib.compress(bytes(raw), level=9)

        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack('!I', len(data)) + tag + data + struct.pack('!I', zlib.crc32(tag + data) & 0xFFFFFFFF)

        png = bytearray(b'\x89PNG\r\n\x1a\n')
        png.extend(chunk(b'IHDR', struct.pack('!IIBBBBB', self.width, self.height, 8, 2, 0, 0, 0)))
        png.extend(chunk(b'IDAT', compressed))
        png.extend(chunk(b'IEND', b''))
        path.write_bytes(bytes(png))

    def save_pdf(self, path: pathlib.Path) -> None:
        width = self.width
        height = self.height
        image_data = zlib.compress(bytes(self.pixels), level=9)
        contents = f'q\n{width} 0 0 {height} 0 0 cm\n/Im0 Do\nQ\n'.encode('ascii')
        objects = [
            b'<< /Type /Catalog /Pages 2 0 R >>',
            b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
            f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>'.encode('ascii'),
            b'<< /Type /XObject /Subtype /Image /Width ' + str(width).encode('ascii') + b' /Height ' + str(height).encode('ascii') + b' /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length ' + str(len(image_data)).encode('ascii') + b' >>\nstream\n' + image_data + b'\nendstream',
            b'<< /Length ' + str(len(contents)).encode('ascii') + b' >>\nstream\n' + contents + b'endstream',
        ]
        pdf = bytearray(b'%PDF-1.4\n')
        offsets = [0]
        for index, obj in enumerate(objects, start=1):
            offsets.append(len(pdf))
            pdf.extend(f'{index} 0 obj\n'.encode('ascii'))
            pdf.extend(obj)
            pdf.extend(b'\nendobj\n')
        xref_start = len(pdf)
        pdf.extend(f'xref\n0 {len(objects) + 1}\n'.encode('ascii'))
        pdf.extend(b'0000000000 65535 f \n')
        for offset in offsets[1:]:
            pdf.extend(f'{offset:010d} 00000 n \n'.encode('ascii'))
        pdf.extend(f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n'.encode('ascii'))
        path.write_bytes(bytes(pdf))


def sanitize_text(text: str) -> str:
    cleaned = []
    for char in text.upper().replace('_', ' '):
        if char in FONT:
            cleaned.append(char)
        elif char in {':', '.', ',', '(', ')', '/', '+', '%'}:
            cleaned.append(' ')
        else:
            cleaned.append(' ')
    return ''.join(cleaned)


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def first_nonempty(row: dict[str, Any], *keys: str, default: str = '') -> str:
    for key in keys:
        value = str(row.get(key, '') or '').strip()
        if value:
            return value
    return default


def parse_wfs(value: str) -> list[str]:
    text = (value or '').strip()
    if not text:
        return []
    if text.startswith('[') and text.endswith(']'):
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                return [str(item).strip() for item in payload if str(item).strip()]
        except json.JSONDecodeError:
            pass
    splitter = '|' if '|' in text else ',' if ',' in text else ';' if ';' in text else None
    if splitter:
        return [item.strip() for item in text.split(splitter) if item.strip()]
    return [text]


def parse_year(row: dict[str, Any]) -> int | None:
    value = first_nonempty(row, 'publication_year', 'year')
    if not value:
        return None
    try:
        year = int(float(value))
    except ValueError:
        return None
    if 1900 <= year <= 2100:
        return year
    return None


def normalize_sample_size(value: str) -> str:
    text = (value or '').strip()
    if not text:
        return 'not_reported'
    compact = text.replace(',', '')
    if compact.lower() == 'not_reported':
        return 'not_reported'
    try:
        return str(int(float(compact)))
    except ValueError:
        return text


def record_core_finding(row: dict[str, Any]) -> str:
    finding = first_nonempty(row, 'core_finding', 'tier_rationale', 'rationale_text')
    if finding:
        return finding
    abstract = first_nonempty(row, 'abstract')
    if not abstract:
        return 'Not reported'
    return textwrap.shorten(abstract, width=180, placeholder='...')


def normalize_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        tier = first_nonempty(row, 'tier')
        crl = first_nonempty(row, 'crl')
        if not tier or not crl:
            continue
        wfs_values = [item for item in parse_wfs(first_nonempty(row, 'wfs')) if item in WFS_ORDER]
        normalized.append({
            'record_id': first_nonempty(row, 'record_id'),
            'title': first_nonempty(row, 'title', default='Untitled study'),
            'abstract': first_nonempty(row, 'abstract'),
            'authors': first_nonempty(row, 'authors', 'author_string', default='Not reported'),
            'year': parse_year(row),
            'journal': first_nonempty(row, 'journal_or_source', 'journal', default='Not reported'),
            'decision': first_nonempty(row, 'decision'),
            'tier': tier,
            'crl': crl,
            'wfs': wfs_values,
            'gi_subsite': first_nonempty(row, 'gi_subsite', default='general_gi'),
            'llm_model': first_nonempty(row, 'llm_model', default='not_reported'),
            'sample_size': normalize_sample_size(first_nonempty(row, 'sample_size', default='not_reported')),
            'comparator': first_nonempty(row, 'comparator', default='not_reported'),
            'core_finding': record_core_finding(row),
        })
    return normalized


def generate_demo_rows(total: int = 50) -> list[dict[str, Any]]:
    random.seed(42)
    tier_pool = ['S'] * 2 + ['I-a'] * 6 + ['I-b'] * 14 + ['II'] * 18 + ['III'] * 10
    crl_by_tier = {
        'S': ['High', 'High', 'Medium'],
        'I-a': ['High', 'Medium', 'Medium'],
        'I-b': ['High', 'Medium', 'Medium', 'Low'],
        'II': ['Medium', 'Medium', 'Low', 'Low'],
        'III': ['Low', 'Low', 'Medium'],
    }
    wfs_by_tier = {
        'S': ['treatment', 'mdt', 'perioperative', 'followup'],
        'I-a': ['diagnosis', 'staging', 'treatment', 'followup'],
        'I-b': ['diagnosis', 'staging', 'treatment', 'research_qc', 'screening'],
        'II': ['diagnosis', 'patient_communication', 'research_qc', 'screening', 'treatment'],
        'III': ['patient_communication', 'research_qc', 'diagnosis'],
    }
    subsites = ['colorectal', 'gastric', 'esophageal', 'small_bowel', 'anal', 'general_gi', 'multiple_gi']
    subsite_weights = [16, 12, 7, 4, 3, 5, 4]
    models = ['GPT-4', 'ChatGPT', 'Claude', 'Gemini', 'LLaMA-2', 'GPT-4o']
    model_weights = [16, 12, 7, 6, 4, 5]
    rows = []
    for index in range(total):
        tier = random.choice(tier_pool)
        crl = random.choice(crl_by_tier[tier])
        wfs_candidates = wfs_by_tier[tier]
        wfs = sorted(set(random.sample(wfs_candidates, k=1 if random.random() < 0.72 else 2)))
        gi_subsite = random.choices(subsites, weights=subsite_weights, k=1)[0]
        llm_model = random.choices(models, weights=model_weights, k=1)[0]
        year = random.choices([2022, 2023, 2024, 2025, 2026], weights=[4, 8, 12, 15, 11], k=1)[0]
        if tier == 'III':
            sample_size = str(random.choice([80, 120, 160, 220, 300]))
            comparator = random.choice(['none', 'residents', 'other_model'])
        elif tier == 'II':
            sample_size = str(random.choice([45, 60, 75, 120, 240, 500]))
            comparator = random.choice(['physicians', 'other_model', 'none'])
        elif tier == 'I-b':
            sample_size = str(random.choice([96, 128, 180, 240, 410, 820]))
            comparator = random.choice(['physicians', 'residents', 'standard_of_care'])
        elif tier == 'I-a':
            sample_size = str(random.choice([55, 72, 108, 160, 220]))
            comparator = random.choice(['physicians', 'multidisciplinary_team', 'standard_of_care'])
        else:
            sample_size = str(random.choice([90, 120, 180]))
            comparator = random.choice(['standard_of_care', 'physicians'])
        rows.append({
            'record_id': f'DEMO{index + 1:03d}',
            'title': f'Demo GI oncology LLM study {index + 1}',
            'abstract': f'Simulated abstract for tier {tier} in {gi_subsite} using {llm_model}.',
            'authors': f'Author {index + 1} et al.',
            'publication_year': str(year),
            'journal_or_source': random.choice(['JCO Clinical Cancer Informatics', 'NPJ Digital Medicine', 'Lancet Digital Health', 'JAMA Network Open']),
            'decision': 'Include',
            'tier': tier,
            'tier_rationale': f'Simulated rationale for {tier}.',
            'crl': crl,
            'crl_rationale': f'Simulated risk rationale for {crl}.',
            'wfs': '|'.join(wfs),
            'gi_subsite': gi_subsite,
            'llm_model': llm_model,
            'sample_size': sample_size,
            'comparator': comparator,
            'labeling_confidence': random.choice(['high', 'medium', 'medium']),
            'core_finding': f'{llm_model} showed simulated performance gains for {gi_subsite} at the {tier} evidence tier.',
        })
    return rows


def save_demo_input(output_dir: pathlib.Path, rows: list[dict[str, Any]]) -> pathlib.Path:
    demo_path = output_dir / 'demo_labeled_records.csv'
    fieldnames = [
        'record_id', 'title', 'abstract', 'authors', 'publication_year', 'journal_or_source', 'decision',
        'tier', 'tier_rationale', 'crl', 'crl_rationale', 'wfs', 'gi_subsite', 'llm_model', 'sample_size',
        'comparator', 'labeling_confidence', 'core_finding'
    ]
    write_csv(demo_path, fieldnames, rows)
    return demo_path


def build_summary_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(records)
    rows: list[dict[str, Any]] = [{'section': 'overall', 'label': 'total_included_studies', 'count': total, 'percent': '100.0'}]
    tier_counts = Counter(record['tier'] for record in records)
    crl_counts = Counter(record['crl'] for record in records)
    subsite_counts = Counter(record['gi_subsite'] for record in records)
    wfs_counts = Counter()
    for record in records:
        for wfs in record['wfs']:
            wfs_counts[wfs] += 1
    for label in TIERS:
        rows.append({'section': 'tier', 'label': label, 'count': tier_counts.get(label, 0), 'percent': f"{(tier_counts.get(label, 0) / total * 100) if total else 0:.1f}"})
    for label in CRL_LEVELS:
        rows.append({'section': 'crl', 'label': label, 'count': crl_counts.get(label, 0), 'percent': f"{(crl_counts.get(label, 0) / total * 100) if total else 0:.1f}"})
    for label in WFS_ORDER:
        rows.append({'section': 'wfs', 'label': label, 'count': wfs_counts.get(label, 0), 'percent': f"{(wfs_counts.get(label, 0) / total * 100) if total else 0:.1f}"})
    for label in GI_SUBSITE_ORDER:
        rows.append({'section': 'gi_subsite', 'label': label, 'count': subsite_counts.get(label, 0), 'percent': f"{(subsite_counts.get(label, 0) / total * 100) if total else 0:.1f}"})
    return rows


def build_high_evidence_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered = [record for record in records if record['tier'] in {'S', 'I-a'}]
    filtered.sort(key=lambda item: (-(item['year'] or 0), item['tier'], item['title']))
    rows = []
    for record in filtered:
        rows.append({
            'authors': record['authors'],
            'year': record['year'] or '',
            'title': record['title'],
            'tier': record['tier'],
            'crl': record['crl'],
            'gi_subsite': record['gi_subsite'],
            'llm_model': record['llm_model'],
            'sample_size': record['sample_size'],
            'comparator': record['comparator'],
            'core_finding': record['core_finding'],
        })
    return rows


def compute_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    years = sorted({record['year'] for record in records if record['year'] is not None})
    if not years:
        years = [2022, 2023, 2024, 2025, 2026]
    if years[0] > 2022:
        years = [2022, 2023, 2024, 2025, 2026]
    tier_counts = Counter(record['tier'] for record in records)
    crl_matrix = {tier: {crl: 0 for crl in CRL_LEVELS} for tier in TIERS}
    wfs_matrix = {tier: {wfs: 0 for wfs in WFS_ORDER} for tier in TIERS}
    subsite_tier = {subsite: {tier: 0 for tier in TIERS} for subsite in GI_SUBSITE_ORDER}
    model_counts = Counter()
    year_tier = {year: {tier: 0 for tier in TIERS} for year in years}
    for record in records:
        tier = record['tier']
        crl = record['crl']
        if tier in TIERS and crl in CRL_LEVELS:
            crl_matrix[tier][crl] += 1
        for wfs in record['wfs']:
            if wfs in WFS_ORDER:
                wfs_matrix[tier][wfs] += 1
        subsite = record['gi_subsite'] if record['gi_subsite'] in GI_SUBSITE_ORDER else 'general_gi'
        subsite_tier[subsite][tier] += 1
        model_counts[record['llm_model'] or 'not_reported'] += 1
        year = record['year']
        if year in year_tier:
            year_tier[year][tier] += 1
    return {
        'total': len(records),
        'tier_counts': tier_counts,
        'crl_matrix': crl_matrix,
        'wfs_matrix': wfs_matrix,
        'subsite_tier': subsite_tier,
        'model_counts': model_counts,
        'year_tier': year_tier,
        'years': years,
    }


def save_report(output_dir: pathlib.Path, title: str, records: list[dict[str, Any]], stats: dict[str, Any], engine: str) -> pathlib.Path:
    report_path = output_dir / 'evidence_map_report.txt'
    top_models = ', '.join(f'{model}:{count}' for model, count in stats['model_counts'].most_common(5)) or 'none'
    lines = [
        title,
        '',
        f'Engine: {engine}',
        f'Total labeled studies: {len(records)}',
        f'Tier counts: {dict(stats["tier_counts"])}',
        f'Top models: {top_models}',
        f'Years covered: {stats["years"]}',
        '',
        'Outputs:',
        '- figure_1_evidence_tier_distribution.(png|pdf)',
        '- figure_2_tier_crl_heatmap.(png|pdf)',
        '- figure_3_tier_wfs_heatmap.(png|pdf)',
        '- figure_4_gi_subsite_distribution.(png|pdf)',
        '- figure_5_llm_model_frequency.(png|pdf)',
        '- figure_6_time_trend.(png|pdf)',
        '- table_1_study_summary.csv',
        '- table_2_high_evidence_studies.csv',
    ]
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path


def save_canvas_outputs(canvas: SimpleCanvas, base_path: pathlib.Path) -> None:
    canvas.save_png(base_path.with_suffix('.png'))
    canvas.save_pdf(base_path.with_suffix('.pdf'))


def draw_title_block(canvas: SimpleCanvas, title: str, subtitle: str) -> None:
    canvas.draw_text(40, 28, title[:52], TEXT_COLOR, scale=3)
    canvas.draw_text(40, 60, subtitle[:70], (80, 80, 80), scale=2)


def draw_axis_frame(canvas: SimpleCanvas, left: int, top: int, width: int, height: int) -> None:
    canvas.draw_line(left, top + height, left + width, top + height, AXIS_COLOR)
    canvas.draw_line(left, top, left, top + height, AXIS_COLOR)
    for fraction in [0.25, 0.5, 0.75]:
        y = top + int(height * (1.0 - fraction))
        canvas.draw_line(left, y, left + width, y, GRID_COLOR)


def draw_label(canvas: SimpleCanvas, x: int, y: int, text: str, scale: int = 2, color: tuple[int, int, int] = TEXT_COLOR) -> None:
    canvas.draw_text(x, y, sanitize_text(text), color, scale=scale)


def render_tier_distribution_basic(base_path: pathlib.Path, title: str, stats: dict[str, Any]) -> None:
    canvas = SimpleCanvas(1500, 950)
    draw_title_block(canvas, 'FIGURE 1 EVIDENCE TIER DISTRIBUTION', sanitize_text(title))
    left, top, width, height = 110, 140, 1260, 650
    draw_axis_frame(canvas, left, top, width, height)
    counts = [stats['tier_counts'].get(tier, 0) for tier in TIERS]
    total = max(1, stats['total'])
    ymax = max(max(counts), 1)
    group = width / len(TIERS)
    bar_w = int(group * 0.45)
    for idx, tier in enumerate(TIERS):
        count = counts[idx]
        bar_h = int(height * count / ymax)
        x = left + int(group * idx + group * 0.28)
        y = top + height - bar_h
        canvas.fill_rect(x, y, bar_w, bar_h, TIER_COLORS[tier])
        canvas.draw_rect(x, y, bar_w, bar_h, AXIS_COLOR)
        draw_label(canvas, x - 4, top + height + 30, tier.replace('-', ''), scale=3)
        draw_label(canvas, x + 8, y - 40, str(count), scale=3)
        draw_label(canvas, x - 6, y - 18, f'{count / total * 100:.0f} PCT', scale=2)
    save_canvas_outputs(canvas, base_path)


def render_heatmap_basic(base_path: pathlib.Path, main_title: str, subtitle: str, rows: list[str], cols: list[str], matrix: dict[str, dict[str, int]], color_fn) -> None:
    cell_w = 120 if len(cols) <= 5 else 95
    canvas = SimpleCanvas(220 + cell_w * len(cols), 250 + 95 * len(rows))
    draw_title_block(canvas, sanitize_text(main_title), sanitize_text(subtitle))
    left, top = 170, 150
    max_value = max((matrix[row][col] for row in rows for col in cols), default=1)
    for col_index, col in enumerate(cols):
        label = WFS_ABBR.get(col, SUBSITE_ABBR.get(col, col)).replace('-', '')
        draw_label(canvas, left + col_index * cell_w + 18, 115, label, scale=2)
    for row_index, row in enumerate(rows):
        draw_label(canvas, 55, top + row_index * 95 + 24, row.replace('-', ''), scale=3)
        for col_index, col in enumerate(cols):
            value = matrix[row][col]
            x = left + col_index * cell_w
            y = top + row_index * 95
            fill = color_fn(value, max_value)
            canvas.fill_rect(x, y, cell_w - 12, 72, fill)
            canvas.draw_rect(x, y, cell_w - 12, 72, AXIS_COLOR)
            text_color = LIGHT_TEXT_COLOR if sum(fill) < 350 else TEXT_COLOR
            draw_label(canvas, x + 24, y + 20, str(value), scale=3, color=text_color)
    save_canvas_outputs(canvas, base_path)


def blend_color(base: tuple[int, int, int], value: int, maximum: int) -> tuple[int, int, int]:
    if maximum <= 0:
        return (245, 245, 245)
    ratio = value / maximum
    white = (247, 247, 247)
    return tuple(int(white[index] + (base[index] - white[index]) * ratio) for index in range(3))


def render_subsite_distribution_basic(base_path: pathlib.Path, title: str, stats: dict[str, Any]) -> None:
    canvas = SimpleCanvas(1750, 980)
    draw_title_block(canvas, 'FIGURE 4 GI SUBSITE DISTRIBUTION', sanitize_text(title))
    left, top, width, height = 110, 150, 1520, 650
    draw_axis_frame(canvas, left, top, width, height)
    ymax = max(sum(stats['subsite_tier'][subsite][tier] for tier in TIERS) for subsite in GI_SUBSITE_ORDER) or 1
    group = width / len(GI_SUBSITE_ORDER)
    bar_w = max(16, int(group * 0.12))
    for subsite_index, subsite in enumerate(GI_SUBSITE_ORDER):
        base_x = left + int(group * subsite_index + group * 0.1)
        draw_label(canvas, base_x - 6, top + height + 28, SUBSITE_ABBR[subsite], scale=2)
        for tier_index, tier in enumerate(TIERS):
            count = stats['subsite_tier'][subsite][tier]
            bar_h = int(height * count / ymax)
            x = base_x + tier_index * (bar_w + 8)
            y = top + height - bar_h
            canvas.fill_rect(x, y, bar_w, bar_h, TIER_COLORS[tier])
            canvas.draw_rect(x, y, bar_w, bar_h, AXIS_COLOR)
    legend_x = 1280
    for idx, tier in enumerate(TIERS):
        canvas.fill_rect(legend_x, 60 + idx * 38, 22, 22, TIER_COLORS[tier])
        draw_label(canvas, legend_x + 34, 60 + idx * 38, tier.replace('-', ''), scale=2)
    save_canvas_outputs(canvas, base_path)


def render_model_frequency_basic(base_path: pathlib.Path, title: str, stats: dict[str, Any]) -> None:
    canvas = SimpleCanvas(1500, 980)
    draw_title_block(canvas, 'FIGURE 5 LLM MODEL FREQUENCY', sanitize_text(title))
    items = stats['model_counts'].most_common(10)
    left, top, width = 280, 150, 1030
    row_h = 70
    max_count = max((count for _, count in items), default=1)
    for idx, (model, count) in enumerate(items):
        y = top + idx * row_h
        draw_label(canvas, 40, y + 16, model[:18], scale=2)
        canvas.fill_rect(left, y + 8, int(width * count / max_count), 36, (61, 90, 128))
        canvas.draw_rect(left, y + 8, width, 36, AXIS_COLOR)
        draw_label(canvas, left + int(width * count / max_count) + 18, y + 16, str(count), scale=2)
    save_canvas_outputs(canvas, base_path)


def render_time_trend_basic(base_path: pathlib.Path, title: str, stats: dict[str, Any]) -> None:
    canvas = SimpleCanvas(1500, 980)
    draw_title_block(canvas, 'FIGURE 6 TIME TREND', sanitize_text(title))
    left, top, width, height = 130, 150, 1220, 650
    draw_axis_frame(canvas, left, top, width, height)
    years = stats['years']
    ymax = max(sum(stats['year_tier'][year][tier] for tier in TIERS) for year in years) or 1
    group = width / len(years)
    bar_w = int(group * 0.45)
    for idx, year in enumerate(years):
        x = left + int(group * idx + group * 0.25)
        y_bottom = top + height
        for tier in TIERS:
            count = stats['year_tier'][year][tier]
            segment = int(height * count / ymax)
            if segment <= 0:
                continue
            y_bottom -= segment
            canvas.fill_rect(x, y_bottom, bar_w, segment, TIER_COLORS[tier])
            canvas.draw_rect(x, y_bottom, bar_w, segment, AXIS_COLOR)
        draw_label(canvas, x - 6, top + height + 28, str(year), scale=2)
    legend_x = 1180
    for idx, tier in enumerate(TIERS):
        canvas.fill_rect(legend_x, 60 + idx * 38, 22, 22, TIER_COLORS[tier])
        draw_label(canvas, legend_x + 34, 60 + idx * 38, tier.replace('-', ''), scale=2)
    save_canvas_outputs(canvas, base_path)


def render_with_matplotlib(output_dir: pathlib.Path, title: str, stats: dict[str, Any]) -> bool:
    if plt is None:
        return False
    plt.rcParams.update({
        'figure.dpi': 300,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })
    try:
        tier_colors = [TIER_COLORS_HEX[tier] for tier in TIERS]

        # Figure 1
        counts = [stats['tier_counts'].get(tier, 0) for tier in TIERS]
        total = max(1, stats['total'])
        fig, ax = plt.subplots(figsize=(8.2, 5.6))
        bars = ax.bar(TIERS, counts, color=tier_colors, edgecolor='#333333', linewidth=0.8)
        ax.set_ylabel('Study count')
        ax.set_title('Figure 1. Evidence Tier distribution\nCounts and proportions among labeled included studies')
        ax.spines[['top', 'right']].set_visible(False)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f'{count}\n{count / total * 100:.1f}%', ha='center', va='bottom', fontsize=9)
        fig.suptitle(title, fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / 'figure_1_evidence_tier_distribution.png', dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / 'figure_1_evidence_tier_distribution.pdf', bbox_inches='tight')
        plt.close(fig)

        # Figure 2
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        matrix = [[stats['crl_matrix'][tier][crl] for crl in CRL_LEVELS] for tier in TIERS]
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(CRL_LEVELS)), CRL_LEVELS)
        ax.set_yticks(range(len(TIERS)), TIERS)
        ax.set_title('Figure 2. Evidence Tier × Clinical Risk Level\nDarker cells indicate more studies')
        for row_index, tier in enumerate(TIERS):
            for col_index, crl in enumerate(CRL_LEVELS):
                ax.text(col_index, row_index, str(stats['crl_matrix'][tier][crl]), ha='center', va='center', color='black', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(title, fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / 'figure_2_tier_crl_heatmap.png', dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / 'figure_2_tier_crl_heatmap.pdf', bbox_inches='tight')
        plt.close(fig)

        # Figure 3
        fig, ax = plt.subplots(figsize=(11.5, 5.6))
        matrix = [[stats['wfs_matrix'][tier][wfs] for wfs in WFS_ORDER] for tier in TIERS]
        im = ax.imshow(matrix, cmap='GnBu', aspect='auto')
        ax.set_xticks(range(len(WFS_ORDER)), [WFS_ABBR[wfs] for wfs in WFS_ORDER], rotation=45, ha='right')
        ax.set_yticks(range(len(TIERS)), TIERS)
        ax.set_title('Figure 3. Evidence Tier × Workflow Stage\nA study may contribute to multiple workflow stages')
        for row_index, tier in enumerate(TIERS):
            for col_index, wfs in enumerate(WFS_ORDER):
                ax.text(col_index, row_index, str(stats['wfs_matrix'][tier][wfs]), ha='center', va='center', color='black', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(title, fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / 'figure_3_tier_wfs_heatmap.png', dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / 'figure_3_tier_wfs_heatmap.pdf', bbox_inches='tight')
        plt.close(fig)

        # Figure 4
        fig, ax = plt.subplots(figsize=(11.5, 5.8))
        x_positions = list(range(len(GI_SUBSITE_ORDER)))
        width = 0.15
        for idx, tier in enumerate(TIERS):
            counts = [stats['subsite_tier'][subsite][tier] for subsite in GI_SUBSITE_ORDER]
            shifted = [x + (idx - 2) * width for x in x_positions]
            ax.bar(shifted, counts, width=width, label=tier, color=TIER_COLORS_HEX[tier], edgecolor='#333333', linewidth=0.6)
        ax.set_xticks(x_positions, [SUBSITE_ABBR[item] for item in GI_SUBSITE_ORDER], rotation=35, ha='right')
        ax.set_ylabel('Study count')
        ax.set_title('Figure 4. GI subsite distribution by Evidence Tier\nGrouped bars summarize translational maturity across GI subsites')
        ax.legend(ncol=5, frameon=False, loc='upper right')
        ax.spines[['top', 'right']].set_visible(False)
        fig.suptitle(title, fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / 'figure_4_gi_subsite_distribution.png', dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / 'figure_4_gi_subsite_distribution.pdf', bbox_inches='tight')
        plt.close(fig)

        # Figure 5
        fig, ax = plt.subplots(figsize=(8.5, 5.6))
        items = stats['model_counts'].most_common(12)
        labels = [item[0] for item in items][::-1]
        values = [item[1] for item in items][::-1]
        ax.barh(labels, values, color='#355C7D', edgecolor='#333333', linewidth=0.6)
        ax.set_xlabel('Study count')
        ax.set_title('Figure 5. Primary LLM model frequency\nMost frequently evaluated models among labeled included studies')
        ax.spines[['top', 'right']].set_visible(False)
        for pos, value in enumerate(values):
            ax.text(value + 0.2, pos, str(value), va='center', fontsize=9)
        fig.suptitle(title, fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / 'figure_5_llm_model_frequency.png', dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / 'figure_5_llm_model_frequency.pdf', bbox_inches='tight')
        plt.close(fig)

        # Figure 6
        fig, ax = plt.subplots(figsize=(8.8, 5.6))
        years = stats['years']
        bottoms = [0] * len(years)
        for tier in TIERS:
            values = [stats['year_tier'][year][tier] for year in years]
            ax.bar(years, values, bottom=bottoms, color=TIER_COLORS_HEX[tier], label=tier, edgecolor='#333333', linewidth=0.5)
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        ax.set_ylabel('Study count')
        ax.set_xlabel('Publication year')
        ax.set_title('Figure 6. Publication-year trend by Evidence Tier\nStacked bars show maturity of evidence over time')
        ax.legend(ncol=5, frameon=False, loc='upper left')
        ax.spines[['top', 'right']].set_visible(False)
        fig.suptitle(title, fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(output_dir / 'figure_6_time_trend.png', dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / 'figure_6_time_trend.pdf', bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception:
        return False


def render_basic_figures(output_dir: pathlib.Path, title: str, stats: dict[str, Any]) -> None:
    render_tier_distribution_basic(output_dir / 'figure_1_evidence_tier_distribution', title, stats)
    render_heatmap_basic(
        output_dir / 'figure_2_tier_crl_heatmap',
        'FIGURE 2 TIER CRL HEATMAP',
        'EVIDENCE TIER BY CLINICAL RISK LEVEL',
        TIERS,
        CRL_LEVELS,
        stats['crl_matrix'],
        lambda value, maximum: blend_color((205, 85, 85), value, maximum),
    )
    render_heatmap_basic(
        output_dir / 'figure_3_tier_wfs_heatmap',
        'FIGURE 3 TIER WFS HEATMAP',
        'EVIDENCE TIER BY WORKFLOW STAGE',
        TIERS,
        WFS_ORDER,
        stats['wfs_matrix'],
        lambda value, maximum: blend_color((69, 117, 180), value, maximum),
    )
    render_subsite_distribution_basic(output_dir / 'figure_4_gi_subsite_distribution', title, stats)
    render_model_frequency_basic(output_dir / 'figure_5_llm_model_frequency', title, stats)
    render_time_trend_basic(output_dir / 'figure_6_time_trend', title, stats)


def main() -> None:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        demo_rows = generate_demo_rows(50)
        demo_input = save_demo_input(output_dir, demo_rows)
        raw_rows = load_rows(demo_input)
    else:
        if not args.input:
            raise SystemExit('--input is required unless --demo is used')
        raw_rows = load_rows(pathlib.Path(args.input))

    records = normalize_records(raw_rows)
    if not records:
        raise SystemExit('No labeled records found; expected non-empty tier and crl columns')

    summary_rows = build_summary_table(records)
    high_evidence_rows = build_high_evidence_table(records)
    write_csv(output_dir / 'table_1_study_summary.csv', ['section', 'label', 'count', 'percent'], summary_rows)
    write_csv(
        output_dir / 'table_2_high_evidence_studies.csv',
        ['authors', 'year', 'title', 'tier', 'crl', 'gi_subsite', 'llm_model', 'sample_size', 'comparator', 'core_finding'],
        high_evidence_rows,
    )

    stats = compute_stats(records)
    rendered = render_with_matplotlib(output_dir, args.title, stats)
    if not rendered:
        render_basic_figures(output_dir, args.title, stats)
    report_path = save_report(output_dir, args.title, records, stats, 'matplotlib' if rendered else 'basic_canvas_fallback')

    print(f'[build_evidence_map] records={len(records)} engine={"matplotlib" if rendered else "basic_canvas_fallback"}')
    print(f'[build_evidence_map] summary_table={output_dir / "table_1_study_summary.csv"}')
    print(f'[build_evidence_map] high_evidence_table={output_dir / "table_2_high_evidence_studies.csv"}')
    print(f'[build_evidence_map] report={report_path}')


if __name__ == '__main__':
    main()
