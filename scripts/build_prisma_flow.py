from __future__ import annotations

import argparse
import json
import pathlib
import struct
import textwrap
import zlib
from typing import Any

TIERS = ['identification', 'screening', 'eligibility', 'included']
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

CANVAS_WIDTH = 1900
CANVAS_HEIGHT = 1360
BOX_FILL = '#EAF3FF'
BOX_BORDER = '#1E4F8C'
BANNER_FILL = '#D3E5FF'
TEXT_COLOR = '#0E2742'
GRID_TEXT = (14, 39, 66)
BOX_FILL_RGB = (234, 243, 255)
BOX_BORDER_RGB = (30, 79, 140)
BANNER_FILL_RGB = (211, 229, 255)
WHITE = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a PRISMA 2020-style flow diagram from screening counts.')
    parser.add_argument('--counts', help='Inline JSON string or path to JSON file with PRISMA node counts')
    parser.add_argument('--output-dir', required=True, help='Output directory for SVG/PNG/JSON')
    parser.add_argument('--demo', action='store_true', help='Use demo counts and render example outputs')
    return parser.parse_args()


def demo_counts() -> dict[str, Any]:
    return {
        'identification': {
            'pubmed': 1204,
            'scopus': 0,
            'other_databases': 0,
            'registers': 0,
            'other_sources': 0,
        },
        'duplicates_removed': 0,
        'records_screened': 1204,
        'records_excluded_title_abstract': 900,
        'reports_sought': 304,
        'reports_not_retrieved': 10,
        'reports_assessed_fulltext': 294,
        'fulltext_exclusions': {
            'duplicate_or_secondary_reports': 32,
            'protocol_b_only_reports': 18,
            'not_llm_or_not_in_scope_or_other_noneligible': 12,
            'preprint_or_source_restricted_from_protocol_a': 2,
        },
        'studies_included_protocol_a': 230,
        'studies_included_protocol_b': 0,
    }


def load_jsonish(value: str) -> Any:
    candidate = pathlib.Path(value)
    if candidate.exists():
        return json.loads(candidate.read_text(encoding='utf-8'))
    return json.loads(value)


def as_int(value: Any, field_name: str) -> int:
    try:
        integer = int(value)
    except Exception as exc:
        raise ValueError(f'Invalid integer for {field_name}: {value!r}') from exc
    if integer < 0:
        raise ValueError(f'{field_name} must be non-negative')
    return integer


def normalize_counts(payload: dict[str, Any]) -> dict[str, Any]:
    screening_payload = payload.get('screening', payload)
    eligibility_payload = payload.get('eligibility', payload)
    included_payload = payload.get('included', payload)
    identification = payload.get('identification', {})
    if not isinstance(identification, dict):
        raise ValueError('counts.identification must be an object')

    pubmed = as_int(identification.get('pubmed', 0), 'identification.pubmed')
    scopus = as_int(identification.get('scopus', 0), 'identification.scopus')
    other_databases = as_int(identification.get('other_databases', 0), 'identification.other_databases')
    registers = as_int(identification.get('registers', 0), 'identification.registers')
    other_sources = as_int(identification.get('other_sources', 0), 'identification.other_sources')

    duplicates_removed = as_int(screening_payload.get('duplicates_removed', payload.get('duplicates_removed', 0)), 'duplicates_removed')
    records_screened = as_int(screening_payload.get('records_screened', payload.get('records_screened', 0)), 'records_screened')
    records_excluded = as_int(
        screening_payload.get('records_excluded_title_abstract', payload.get('records_excluded_title_abstract', 0)),
        'records_excluded_title_abstract',
    )
    reports_sought = as_int(screening_payload.get('reports_sought', payload.get('reports_sought', 0)), 'reports_sought')
    reports_not_retrieved = as_int(
        screening_payload.get('reports_not_retrieved', payload.get('reports_not_retrieved', 0)),
        'reports_not_retrieved',
    )
    reports_assessed = as_int(
        eligibility_payload.get('reports_assessed_fulltext', payload.get('reports_assessed_fulltext', 0)),
        'reports_assessed_fulltext',
    )
    protocol_a = as_int(
        included_payload.get('studies_included_protocol_a', payload.get('studies_included_protocol_a', 0)),
        'studies_included_protocol_a',
    )
    protocol_b = as_int(
        included_payload.get('studies_included_protocol_b', payload.get('studies_included_protocol_b', 0)),
        'studies_included_protocol_b',
    )

    fulltext_exclusions_obj = eligibility_payload.get('fulltext_exclusions', payload.get('fulltext_exclusions', {}))
    if not isinstance(fulltext_exclusions_obj, dict):
        raise ValueError('fulltext_exclusions must be an object')
    fulltext_exclusions = {key: as_int(value, f'fulltext_exclusions.{key}') for key, value in fulltext_exclusions_obj.items()}
    exclusion_total = sum(fulltext_exclusions.values())

    other_methods = payload.get('other_methods', {})
    if other_methods is None:
        other_methods = {}
    if not isinstance(other_methods, dict):
        raise ValueError('other_methods must be an object if provided')
    other_reports_sought = as_int(other_methods.get('reports_sought', other_sources), 'other_methods.reports_sought')
    other_reports_not_retrieved = as_int(other_methods.get('reports_not_retrieved', 0), 'other_methods.reports_not_retrieved')
    other_reports_assessed = as_int(
        other_methods.get('reports_assessed_fulltext', max(other_reports_sought - other_reports_not_retrieved, 0)),
        'other_methods.reports_assessed_fulltext',
    )

    database_total = pubmed + scopus + other_databases + registers
    total_identified = database_total + other_sources
    included_total = protocol_a + protocol_b

    warnings: list[str] = []
    expected_screened = max(database_total - duplicates_removed, 0)
    if records_screened != expected_screened:
        warnings.append(
            f'records_screened ({records_screened}) does not equal database_total - duplicates_removed ({expected_screened})'
        )
    expected_reports_assessed = max(reports_sought - reports_not_retrieved, 0)
    if reports_assessed != expected_reports_assessed:
        warnings.append(
            f'reports_assessed_fulltext ({reports_assessed}) does not equal reports_sought - reports_not_retrieved ({expected_reports_assessed})'
        )
    expected_protocol_a_included = max(reports_assessed - exclusion_total, 0)
    if protocol_a != expected_protocol_a_included:
        warnings.append(
            f'studies_included_protocol_a ({protocol_a}) does not equal reports_assessed_fulltext - fulltext_exclusions_total ({expected_protocol_a_included})'
        )
    if other_sources > 0 and 'other_methods' not in payload:
        warnings.append('other_sources > 0 but detailed other_methods counts were not provided; right-column retrieval nodes were imputed from other_sources')

    return {
        'template_variant': 'PRISMA_2020_new_review_databases_and_other_methods',
        'input': payload,
        'identification': {
            'pubmed': pubmed,
            'scopus': scopus,
            'other_databases': other_databases,
            'registers': registers,
            'database_total': database_total,
            'other_sources': other_sources,
            'total_identified': total_identified,
        },
        'screening': {
            'duplicates_removed': duplicates_removed,
            'records_screened': records_screened,
            'records_excluded_title_abstract': records_excluded,
            'reports_sought': reports_sought,
            'reports_not_retrieved': reports_not_retrieved,
        },
        'eligibility': {
            'reports_assessed_fulltext': reports_assessed,
            'fulltext_exclusions': fulltext_exclusions,
            'fulltext_exclusions_total': exclusion_total,
        },
        'included': {
            'studies_included_protocol_a': protocol_a,
            'studies_included_protocol_b': protocol_b,
            'studies_included_total': included_total,
        },
        'other_methods': {
            'other_sources': other_sources,
            'reports_sought': other_reports_sought,
            'reports_not_retrieved': other_reports_not_retrieved,
            'reports_assessed_fulltext': other_reports_assessed,
        },
        'validation_warnings': warnings,
    }


def sanitize_text(text: str) -> str:
    cleaned = []
    for char in text.upper().replace('_', ' '):
        if char in FONT:
            cleaned.append(char)
        elif char in {':', '.', ',', '(', ')', '/', '+', '=', '*'}:
            cleaned.append(' ')
        else:
            cleaned.append(' ')
    return ''.join(cleaned)


def wrap_text(text: str, width: int) -> list[str]:
    parts = []
    for paragraph in text.split('\n'):
        stripped = paragraph.strip()
        if not stripped:
            parts.append('')
            continue
        parts.extend(textwrap.wrap(stripped, width=width, break_long_words=False, break_on_hyphens=False) or [''])
    return parts


class SimpleCanvas:
    def __init__(self, width: int, height: int, bg: tuple[int, int, int] = WHITE) -> None:
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

    def draw_text(self, x: int, y: int, text: str, color: tuple[int, int, int] = GRID_TEXT, scale: int = 2) -> None:
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


def draw_box_png(canvas: SimpleCanvas, spec: dict[str, Any]) -> None:
    fill = spec.get('fill_rgb', BOX_FILL_RGB)
    border = spec.get('border_rgb', BOX_BORDER_RGB)
    x = spec['x']
    y = spec['y']
    w = spec['w']
    h = spec['h']
    canvas.fill_rect(x, y, w, h, fill)
    canvas.draw_rect(x, y, w, h, border)
    lines = []
    for line in spec['lines']:
        lines.extend(wrap_text(line, spec.get('wrap', 28)))
    scale = spec.get('scale', 2)
    line_height = 9 * scale
    total_height = len(lines) * line_height
    cursor_y = y + max(10, (h - total_height) // 2)
    for line in lines:
        line_width = len(sanitize_text(line)) * 6 * scale
        cursor_x = x + max(10, (w - line_width) // 2)
        canvas.draw_text(cursor_x, cursor_y, line, spec.get('text_rgb', GRID_TEXT), scale=scale)
        cursor_y += line_height


def draw_arrow_png(canvas: SimpleCanvas, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int] = BOX_BORDER_RGB) -> None:
    x0, y0 = start
    x1, y1 = end
    canvas.draw_line(x0, y0, x1, y1, color)
    if x0 == x1:
        direction = 1 if y1 >= y0 else -1
        canvas.draw_line(x1, y1, x1 - 8, y1 - 12 * direction, color)
        canvas.draw_line(x1, y1, x1 + 8, y1 - 12 * direction, color)
    elif y0 == y1:
        direction = 1 if x1 >= x0 else -1
        canvas.draw_line(x1, y1, x1 - 12 * direction, y1 - 8, color)
        canvas.draw_line(x1, y1, x1 - 12 * direction, y1 + 8, color)


def build_layout(counts: dict[str, Any]) -> dict[str, Any]:
    identification = counts['identification']
    screening = counts['screening']
    eligibility = counts['eligibility']
    included = counts['included']
    other_methods = counts['other_methods']
    show_other_methods = any(
        int(other_methods.get(key, 0) or 0) > 0
        for key in ['other_sources', 'reports_sought', 'reports_not_retrieved', 'reports_assessed_fulltext']
    )
    included_total = included["studies_included_total"] if show_other_methods else included["studies_included_protocol_a"]
    exclusion_map = {
        'duplicate_or_secondary_reports': 'Duplicate or secondary reports',
        'protocol_b_only_reports': 'Supplementary-only reports',
        'not_llm_or_not_in_scope_or_other_noneligible': 'Not LLM / out of scope / other noneligible',
        'preprint_or_source_restricted_from_protocol_a': 'Non-primary or source-restricted',
        'reclassified_from_core_primary_denominator': 'Publication-form exclusions',
        'E1_not_llm': 'E1 Not LLM',
        'E2_not_gi_onc': 'E2 Not GI oncology',
        'E3_review': 'E3 Review/editorial',
        'E4_no_data': 'E4 No original data',
        'E5_abstract_only': 'E5 Abstract only',
        'E6_duplicate': 'E6 Duplicate report',
        'E7_before_2022': 'E7 Before 2022',
    }
    preferred_exclusion_order = [
        'duplicate_or_secondary_reports',
        'protocol_b_only_reports',
        'not_llm_or_not_in_scope_or_other_noneligible',
        'preprint_or_source_restricted_from_protocol_a',
        'reclassified_from_core_primary_denominator',
        'E1_not_llm',
        'E2_not_gi_onc',
        'E3_review',
        'E4_no_data',
        'E5_abstract_only',
        'E6_duplicate',
        'E7_before_2022',
    ]
    exclusion_lines = ['Reports excluded:']
    for key in preferred_exclusion_order:
        value = eligibility['fulltext_exclusions'].get(key)
        if value:
            exclusion_lines.append(f'{exclusion_map.get(key, key.replace("_", " "))} (n = {value})')
    if len(exclusion_lines) == 1:
        for key, value in eligibility['fulltext_exclusions'].items():
            exclusion_lines.append(f'{exclusion_map.get(key, key.replace("_", " "))} (n = {value})')

    stage_banners = [
        {'id': 'banner_identification', 'x': 36, 'y': 96, 'w': 120, 'h': 320, 'lines': ['IDENTI', 'FICATION'], 'fill': BANNER_FILL, 'border': BOX_BORDER, 'fill_rgb': BANNER_FILL_RGB, 'border_rgb': BOX_BORDER_RGB, 'scale': 2, 'wrap': 18},
        {'id': 'banner_screening', 'x': 36, 'y': 436, 'w': 120, 'h': 188, 'lines': ['SCREEN', 'ING'], 'fill': BANNER_FILL, 'border': BOX_BORDER, 'fill_rgb': BANNER_FILL_RGB, 'border_rgb': BOX_BORDER_RGB, 'scale': 2, 'wrap': 18},
        {'id': 'banner_eligibility', 'x': 36, 'y': 648, 'w': 120, 'h': 324, 'lines': ['ELIGI', 'BILITY'], 'fill': BANNER_FILL, 'border': BOX_BORDER, 'fill_rgb': BANNER_FILL_RGB, 'border_rgb': BOX_BORDER_RGB, 'scale': 2, 'wrap': 18},
        {'id': 'banner_included', 'x': 36, 'y': 1000, 'w': 120, 'h': 180, 'lines': ['INCLU', 'DED'], 'fill': BANNER_FILL, 'border': BOX_BORDER, 'fill_rgb': BANNER_FILL_RGB, 'border_rgb': BOX_BORDER_RGB, 'scale': 2, 'wrap': 18},
    ]

    boxes = [
        {
            'id': 'db_identified', 'x': 210, 'y': 120, 'w': 360, 'h': 130,
            'lines': [
                'Records identified from:',
                f'PubMed (n = {identification["pubmed"]})',
                f'Scopus (n = {identification["scopus"]})',
                f'Other databases (n = {identification["other_databases"]})',
                f'Registers (n = {identification["registers"]})',
            ],
            'wrap': 26,
        },
        {
            'id': 'removed_before_screening', 'x': 620, 'y': 120, 'w': 330, 'h': 130,
            'lines': [
                'Records removed before screening:',
                f'Duplicate records removed (n = {screening["duplicates_removed"]})',
                'Automation tools excluded (n = 0)',
                'Other reasons (n = 0)',
            ],
            'wrap': 26,
        },
        {
            'id': 'records_screened', 'x': 210, 'y': 320, 'w': 360, 'h': 88,
            'lines': [f'Records screened (n = {screening["records_screened"]})'], 'wrap': 24,
        },
        {
            'id': 'records_excluded', 'x': 620, 'y': 320, 'w': 330, 'h': 88,
            'lines': [f'Records excluded (n = {screening["records_excluded_title_abstract"]})'], 'wrap': 24,
        },
        {
            'id': 'reports_sought', 'x': 210, 'y': 470, 'w': 360, 'h': 88,
            'lines': [f'Reports sought for retrieval (n = {screening["reports_sought"]})'], 'wrap': 26,
        },
        {
            'id': 'reports_not_retrieved', 'x': 620, 'y': 470, 'w': 330, 'h': 88,
            'lines': [f'Reports not retrieved (n = {screening["reports_not_retrieved"]})'], 'wrap': 25,
        },
        {
            'id': 'reports_assessed', 'x': 210, 'y': 620, 'w': 360, 'h': 94,
            'lines': [f'Reports assessed for eligibility (n = {eligibility["reports_assessed_fulltext"]})'], 'wrap': 25,
        },
        {
            'id': 'fulltext_exclusions', 'x': 620, 'y': 590, 'w': 330, 'h': 220,
            'lines': exclusion_lines,
            'wrap': 27,
        },
        {
            'id': 'included', 'x': 700, 'y': 1060, 'w': 520, 'h': 120,
            'lines': [
                f'Studies included in review (n = {included_total})',
                f'Main peer-reviewed studies (n = {included["studies_included_protocol_a"]})',
            ],
            'wrap': 34,
        },
    ]
    if show_other_methods:
        boxes.extend(
            [
                {
                    'id': 'other_identified', 'x': 1110, 'y': 120, 'w': 360, 'h': 130,
                    'lines': [
                        'Records identified from other methods:',
                        f'Other sources (n = {other_methods["other_sources"]})',
                    ],
                    'wrap': 28,
                },
                {
                    'id': 'other_reports_sought', 'x': 1110, 'y': 360, 'w': 360, 'h': 88,
                    'lines': [f'Reports sought for retrieval (n = {other_methods["reports_sought"]})'], 'wrap': 26,
                },
                {
                    'id': 'other_reports_not_retrieved', 'x': 1500, 'y': 360, 'w': 300, 'h': 88,
                    'lines': [f'Reports not retrieved (n = {other_methods["reports_not_retrieved"]})'], 'wrap': 24,
                },
                {
                    'id': 'other_reports_assessed', 'x': 1110, 'y': 510, 'w': 360, 'h': 94,
                    'lines': [f'Reports assessed for eligibility (n = {other_methods["reports_assessed_fulltext"]})'], 'wrap': 25,
                },
            ]
        )
        boxes[-5]['lines'].append(f'Supplementary studies summarized separately (n = {included["studies_included_protocol_b"]})')

    arrows = [
        {'start': (390, 250), 'end': (390, 320)},
        {'start': (570, 185), 'end': (620, 185)},
        {'start': (570, 364), 'end': (620, 364)},
        {'start': (390, 408), 'end': (390, 470)},
        {'start': (570, 514), 'end': (620, 514)},
        {'start': (390, 558), 'end': (390, 620)},
        {'start': (570, 667), 'end': (620, 667)},
        {'start': (390, 714), 'end': (390, 880)},
        {'start': (390, 880), 'end': (960, 880)},
        {'start': (960, 880), 'end': (960, 1060)},
    ]
    if show_other_methods:
        arrows.extend(
            [
                {'start': (1290, 250), 'end': (1290, 360)},
                {'start': (1470, 404), 'end': (1500, 404)},
                {'start': (1290, 448), 'end': (1290, 510)},
                {'start': (1290, 604), 'end': (1290, 880)},
                {'start': (1290, 880), 'end': (960, 880)},
            ]
        )

    headings = [
        {'text': 'Identification of studies via databases and registers', 'x': 390, 'y': 88},
        {'text': 'PRISMA 2020 flow diagram', 'x': 960, 'y': 36, 'size': 26, 'weight': 'bold'},
        {'text': 'Page MJ et al. BMJ 2021 layout adapted for protocol-tracked LLM review accounting', 'x': 960, 'y': 64, 'size': 14},
    ]
    if show_other_methods:
        headings.insert(1, {'text': 'Identification of studies via other methods', 'x': 1290, 'y': 88})
    footnotes = [
        'Optional automation-removal boxes are shown as zero when unavailable.',
        'Right-column other-method retrieval counts are imputed from other_sources unless provided explicitly.',
    ]
    return {'stage_banners': stage_banners, 'boxes': boxes, 'arrows': arrows, 'headings': headings, 'footnotes': footnotes}


def svg_escape(text: str) -> str:
    return (
        text.replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
    )


def render_svg(layout: dict[str, Any], path: pathlib.Path) -> None:
    elements: list[str] = []
    elements.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_WIDTH}" height="{CANVAS_HEIGHT}" viewBox="0 0 {CANVAS_WIDTH} {CANVAS_HEIGHT}">')
    elements.append('<rect width="100%" height="100%" fill="#FFFFFF"/>')

    for heading in layout['headings']:
        anchor = 'middle'
        elements.append(
            f'<text x="{heading["x"]}" y="{heading["y"]}" text-anchor="{anchor}" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{heading.get("size", 18)}" '
            f'font-weight="{heading.get("weight", "600")}" fill="{TEXT_COLOR}">{svg_escape(heading["text"])}</text>'
        )

    def emit_box(spec: dict[str, Any]) -> None:
        elements.append(
            f'<rect x="{spec["x"]}" y="{spec["y"]}" width="{spec["w"]}" height="{spec["h"]}" '
            f'rx="8" ry="8" fill="{spec.get("fill", BOX_FILL)}" stroke="{spec.get("border", BOX_BORDER)}" stroke-width="2.5"/>'
        )
        wrapped = []
        for line in spec['lines']:
            wrapped.extend(wrap_text(line, spec.get('wrap', 30)))
        line_height = spec.get('line_height', 20)
        start_y = spec['y'] + max(22, (spec['h'] - line_height * max(1, len(wrapped))) / 2 + 12)
        elements.append(
            f'<text x="{spec["x"] + spec["w"] / 2}" y="{start_y}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="{spec.get("font_size", 16)}" fill="{TEXT_COLOR}">'
        )
        for index, line in enumerate(wrapped):
            dy = 0 if index == 0 else line_height
            elements.append(f'<tspan x="{spec["x"] + spec["w"] / 2}" dy="{dy}">{svg_escape(line)}</tspan>')
        elements.append('</text>')

    for spec in layout['stage_banners']:
        emit_box(spec)
    for spec in layout['boxes']:
        emit_box(spec)

    elements.append('<defs><marker id="arrowhead" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto"><polygon points="0 0, 10 4, 0 8" fill="#1E4F8C"/></marker></defs>')
    for arrow in layout['arrows']:
        x0, y0 = arrow['start']
        x1, y1 = arrow['end']
        elements.append(
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{BOX_BORDER}" stroke-width="2.5" marker-end="url(#arrowhead)"/>'
        )

    foot_y = 1245
    for footnote in layout['footnotes']:
        elements.append(
            f'<text x="210" y="{foot_y}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="{TEXT_COLOR}">• {svg_escape(footnote)}</text>'
        )
        foot_y += 22

    elements.append('</svg>')
    path.write_text('\n'.join(elements) + '\n', encoding='utf-8')


def render_png(layout: dict[str, Any], path: pathlib.Path) -> None:
    canvas = SimpleCanvas(CANVAS_WIDTH, CANVAS_HEIGHT)
    for spec in layout['stage_banners']:
        draw_box_png(canvas, spec)
    for spec in layout['boxes']:
        draw_box_png(canvas, spec)
    for arrow in layout['arrows']:
        draw_arrow_png(canvas, arrow['start'], arrow['end'])
    for heading in layout['headings']:
        scale = 3 if heading.get('size', 18) >= 22 else 2
        text = heading['text']
        line_width = len(sanitize_text(text)) * 6 * scale
        canvas.draw_text(max(20, heading['x'] - line_width // 2), heading['y'] - 10, text, GRID_TEXT, scale=scale)
    foot_y = 1245
    for footnote in layout['footnotes']:
        canvas.draw_text(210, foot_y, footnote, GRID_TEXT, scale=2)
        foot_y += 22
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save_png(path)


def main() -> None:
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        counts_payload = demo_counts()
    else:
        if not args.counts:
            raise SystemExit('--counts is required unless --demo is used')
        counts_payload = load_jsonish(args.counts)
        if not isinstance(counts_payload, dict):
            raise SystemExit('counts payload must be a JSON object')

    normalized = normalize_counts(counts_payload)
    layout = build_layout(normalized)

    json_path = output_dir / 'prisma_numbers.json'
    svg_path = output_dir / 'prisma_flow.svg'
    png_path = output_dir / 'prisma_flow.png'

    json_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    render_svg(layout, svg_path)
    render_png(layout, png_path)

    print(f'[build_prisma_flow] json={json_path}')
    print(f'[build_prisma_flow] svg={svg_path}')
    print(f'[build_prisma_flow] png={png_path}')
    if normalized['validation_warnings']:
        for warning in normalized['validation_warnings']:
            print(f'[build_prisma_flow] warning={warning}')


if __name__ == '__main__':
    main()
