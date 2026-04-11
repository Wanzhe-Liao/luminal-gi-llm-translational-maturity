from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import textwrap
from collections import Counter, defaultdict
from html import escape
from typing import Any

from generate_manuscript_figures import (
    STYLE_PRESETS,
    SUBSITE_COLORS,
    TRIPOD_ITEM_LABELS,
    classify_model,
    find_tier_csv,
    load_csv,
    load_json,
    load_tripod_item_rates,
    normalize_records,
)
from pipeline_lib import CRL_LEVELS, GI_SUBSITE_ORDER, TIERS, WFS_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate HTML-styled SVG manuscript figures.')
    parser.add_argument('--input-dir', required=True, help='Analysis output root directory')
    parser.add_argument('--figures-dir', help='Output directory for main figure SVG/HTML assets')
    parser.add_argument('--supplementary-dir', help='Output directory for supplementary figure SVG/HTML assets')
    parser.add_argument('--style', default='nature', choices=sorted(STYLE_PRESETS.keys()), help='Visual style preset')
    return parser.parse_args()


def short_label(value: str, limit: int = 58) -> str:
    text = ' '.join((value or '').split())
    return textwrap.shorten(text, width=limit, placeholder='…') if text else ''


def citation_label(title: str, year: str | int | None = None, first_author: str | None = None, limit: int = 28) -> str:
    """Author et al. (year) if first_author available, else truncated title."""
    if first_author and first_author.strip():
        name = first_author.strip()
        return f'{name} et al. ({year})' if year else f'{name} et al.'
    text = ' '.join((title or '').split())
    short = textwrap.shorten(text, width=limit, placeholder='…') if text else ''
    if year:
        return f'{short} ({year})'
    return short


def forest_study_label(row: dict[str, Any]) -> str:
    first_author = re.sub(r'\s*\(\d+\)\s*', ' ', (row.get('first_author') or '')).strip()
    first_author = ' '.join(first_author.split())
    if first_author and first_author.lower() != 'not reported':
        surname = first_author.split(',')[0].strip() if ',' in first_author else first_author.split()[-1]
        year = row.get('publication_year')
        return f'{surname} et al. ({year})' if year else f'{surname} et al.'
    return citation_label(row.get('study_label', ''), row.get('publication_year'))


def ecosystem_model_family_label(model_name: str) -> str:
    normalized = ' '.join((model_name or '').split())
    lowered = normalized.lower()
    if not lowered or lowered in {'not_reported', 'not reported', 'other', 'multiple'}:
        return 'Not reported'
    if (
        'chatgpt' in lowered
        or lowered == 'gpt'
        or any(token in lowered for token in ['gpt-4', 'gpt4', 'gpt-5', 'gpt5', 'gpt-3.5', 'gpt3.5', 'gpt-4.5', 'gpt4.5'])
    ):
        return 'ChatGPT / GPT family'
    if 'llama' in lowered:
        return 'LLaMA family'
    if 'deepseek' in lowered:
        return 'DeepSeek family'
    if 'gemini' in lowered:
        return 'Gemini family'
    if 'claude' in lowered:
        return 'Claude family'
    if 'gemma' in lowered:
        return 'Gemma family'
    return normalized


def wrap_label(value: str, width: int = 20) -> list[str]:
    text = ' '.join((value or '').split())
    return textwrap.wrap(text, width=width) or ['']


def fmt_pct(value: float) -> str:
    return f'{value:.1f}%'


def fmt_ci(mid: float, low: float, high: float) -> str:
    return f'{mid:.3f} [{low:.3f}, {high:.3f}]'


def fmt_p_value(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return 'P = NA'
    if value < 0.001:
        return 'P < 0.001'
    return f'P = {value:.3f}'


def tier_s_absent_note(records: list[dict[str, Any]]) -> str:
    return '; Tier S absent (n=0)' if all(record.get('tier') != 'S' for record in records) else ''


def heterogeneity_note(pooled_result: dict[str, Any] | None) -> str:
    if not pooled_result:
        return ''
    heterogeneity = pooled_result.get('heterogeneity') or {}
    if not heterogeneity:
        return ''
    segments: list[str] = []
    i2 = heterogeneity.get('I2')
    if i2 is not None:
        segments.append(f"I²={safe_float_num(i2):.1f}%")
    q_p = heterogeneity.get('Q_p_value')
    if q_p is not None:
        segments.append(f"Q, {fmt_p_value(safe_float_num(q_p))}")
    pi_low = heterogeneity.get('prediction_interval_lower')
    pi_high = heterogeneity.get('prediction_interval_upper')
    if pi_low is not None and pi_high is not None:
        segments.append(f"PI {safe_float_num(pi_low):.2f}-{safe_float_num(pi_high):.2f}")
    return ' · '.join(segments)


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    raw = hex_color.lstrip('#')
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def blend(color_a: str, color_b: str, weight: float) -> str:
    red_a, green_a, blue_a = hex_to_rgb(color_a)
    red_b, green_b, blue_b = hex_to_rgb(color_b)
    weight = max(0.0, min(1.0, weight))
    red = round(red_a + (red_b - red_a) * weight)
    green = round(green_a + (green_b - green_a) * weight)
    blue = round(blue_a + (blue_b - blue_a) * weight)
    return f'#{red:02X}{green:02X}{blue:02X}'


def text_block(
    x: float,
    y: float,
    lines: list[str],
    *,
    css_class: str = 'body',
    anchor: str = 'start',
    line_height: int = 22,
    fill: str | None = None,
    weight: int | None = None,
    size: int | None = None,
) -> str:
    attrs = [f'x="{x}"', f'y="{y}"', f'class="{css_class}"', f'text-anchor="{anchor}"']
    if fill:
        attrs.append(f'fill="{fill}"')
    if weight:
        attrs.append(f'font-weight="{weight}"')
    if size:
        attrs.append(f'font-size="{size}"')
    tspans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else line_height
        tspans.append(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    return f'<text {" ".join(attrs)}>{"".join(tspans)}</text>'


def rounded_box(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str,
    radius: int = 18,
    opacity: float = 1.0,
    extra: str = '',
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
        f'fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="1.25"{extra} />'
    )


def circle(cx: float, cy: float, radius: float, fill: str, stroke: str = 'none', stroke_width: float = 0) -> str:
    return f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" />'


def rect(x: float, y: float, width: float, height: float, fill: str, stroke: str = 'none', stroke_width: float = 0) -> str:
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" />'


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    width: float = 2,
    dash: str | None = None,
    marker_end: bool = False,
    opacity: float = 1.0,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ''
    marker_attr = ' marker-end="url(#arrow)"' if marker_end else ''
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}" '
        f'stroke-linecap="round" opacity="{opacity}"{dash_attr}{marker_attr} />'
    )


def polyline(points: list[tuple[float, float]], *, stroke: str, width: float = 3, fill: str = 'none', opacity: float = 1.0) -> str:
    point_string = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
    return (
        f'<polyline points="{point_string}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}" />'
    )


def polygon(points: list[tuple[float, float]], *, fill: str, stroke: str = 'none', width: float = 0, opacity: float = 1.0) -> str:
    point_string = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
    return f'<polygon points="{point_string}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}" />'


def svg_document(width: int, height: int, title: str, subtitle: str, body: str, palette: dict[str, str], *, max_display_width: int = 960) -> str:
    if width > max_display_width:
        scale = max_display_width / width
        disp_w = max_display_width
        disp_h = int(height * scale)
    else:
        disp_w = width
        disp_h = height
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{disp_w}" height="{disp_h}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="figureTitle figureSubtitle">
  <title id="figureTitle">{escape(title)}</title>
  <desc id="figureSubtitle">{escape(subtitle)}</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{palette['gray']}"/>
    </marker>
    <filter id="softShadow" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#16324f" flood-opacity="0.05"/>
    </filter>
    <style><![CDATA[
      .bg {{ fill: #ffffff; }}
      .title {{ font: 700 26px 'Helvetica Neue', Arial, sans-serif; fill: {palette['text']}; letter-spacing: -0.3px; }}
      .subtitle {{ font: 400 14px 'Helvetica Neue', Arial, sans-serif; fill: {palette['text']}; }}
      .note {{ font: italic 400 12px 'Helvetica Neue', Arial, sans-serif; fill: {palette['text']}; }}
      .axis {{ font: 600 12px 'Helvetica Neue', Arial, sans-serif; fill: {palette['text']}; }}
      .label {{ font: 500 13px 'Helvetica Neue', Arial, sans-serif; fill: {palette['text']}; }}
      .body {{ font: 400 13px 'Helvetica Neue', Arial, sans-serif; fill: {palette['text']}; }}
      .small {{ font: 400 11px 'Helvetica Neue', Arial, sans-serif; fill: {palette['text']}; }}
      .tiny {{ font: 400 10px 'Helvetica Neue', Arial, sans-serif; fill: {palette['text']}; }}
      .grid {{ stroke: {palette['grid']}; stroke-width: 0.75; stroke-dasharray: 3 5; opacity: 0.7; }}
      .tick {{ stroke: {palette['grid']}; stroke-width: 0.75; }}
      .card {{ filter: url(#softShadow); }}
    ]]></style>
  </defs>
  <rect class="bg" x="0" y="0" width="{width}" height="{height}" />
  {body}
</svg>
'''


def html_document(title: str, subtitle: str, svg_markup: str) -> str:
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #eef3f9;
      --card: #ffffff;
      --ink: #1f2d3d;
      --muted: #5b6470;
      --border: #d6dce5;
    }}
    html, body {{ margin: 0; padding: 0; background: linear-gradient(180deg, var(--bg), #f8fbff); font-family: Arial, Helvetica, sans-serif; color: var(--ink); }}
    .frame {{ max-width: 960px; margin: 0 auto; padding: 28px; }}
    .shell {{ background: var(--card); border: 1px solid var(--border); border-radius: 18px; box-shadow: 0 12px 26px rgba(31,45,61,0.05); overflow: hidden; }}
    svg {{ width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
  <div class="frame">
    <div class="shell">
      {svg_markup}
    </div>
  </div>
</body>
</html>
'''


def save_dual_output(path_without_suffix: pathlib.Path, title: str, subtitle: str, svg_markup: str) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    svg_path = path_without_suffix.with_suffix('.svg')
    html_path = path_without_suffix.with_suffix('.html')
    svg_path.write_text(svg_markup, encoding='utf-8')
    html_path.write_text(html_document(title, subtitle, svg_markup), encoding='utf-8')


def scale_x(value: float, domain_min: float, domain_max: float, left: float, right: float) -> float:
    if math.isclose(domain_max, domain_min):
        return left
    return left + (value - domain_min) / (domain_max - domain_min) * (right - left)


def scale_y(value: float, domain_min: float, domain_max: float, bottom: float, top: float) -> float:
    if math.isclose(domain_max, domain_min):
        return bottom
    return bottom - (value - domain_min) / (domain_max - domain_min) * (bottom - top)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value or '').replace(',', '').strip()))
    except Exception:
        return default


def safe_float_num(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value or '').replace(',', '').strip())
    except Exception:
        return default


def pct_label(value: float, digits: int = 1) -> str:
    return f'{value:.{digits}f}%'


def family_label(value: str) -> str:
    mapping = {
        'accuracy': 'Accuracy',
        'sensitivity': 'Sensitivity',
        'auc_auroc': 'AUC / AUROC',
        'f1': 'F1 score',
    }
    return mapping.get(value, value.replace('_', ' ').title())


def load_yearly_maturity_summary(input_dir: pathlib.Path) -> list[dict[str, Any]]:
    rows = load_csv(input_dir / 'statistics' / 'maturity_yearly_summary.csv')
    normalized = []
    for row in rows:
        normalized.append(
            {
                'year': safe_int(row.get('year')),
                'study_count': safe_int(row.get('study_count')),
                'tier_mean_rank': safe_float_num(row.get('tier_mean_rank')),
                'high_evidence_share_percent': safe_float_num(row.get('high_evidence_share_percent')),
                'tripod_core_mean_score': safe_float_num(row.get('tripod_core_mean_score')),
                'tripod_core_mean_percent': safe_float_num(row.get('tripod_core_mean_percent')),
                'use_case_count': safe_int(row.get('use_case_count')),
                'readiness_mean_rank': safe_float_num(row.get('readiness_mean_rank')),
                'readiness_higher_stage_share_percent': safe_float_num(row.get('readiness_higher_stage_share_percent')),
                'not_ready_use_case_count': safe_int(row.get('not_ready_use_case_count')),
                'external_validation_use_case_count': safe_int(row.get('external_validation_use_case_count')),
                'human_review_use_case_count': safe_int(row.get('human_review_use_case_count')),
                'prospective_trial_use_case_count': safe_int(row.get('prospective_trial_use_case_count')),
            }
        )
    return sorted(normalized, key=lambda row: row['year'])


def figure_one_prisma(input_dir: pathlib.Path, palette: dict[str, str]) -> tuple[str, str, str]:
    """PRISMA 2020 flow diagram — academic journal style with stage-coloured bands."""
    prisma = load_json(input_dir / 'prisma' / 'prisma_numbers.json')
    identification = prisma['identification']
    screening = prisma['screening']
    eligibility = prisma['eligibility']
    included = prisma['included']
    other_methods = prisma.get('other_methods', {})

    # ── Build exclusion-reason list (handles both key-name conventions) ──
    raw_excl = eligibility.get('fulltext_exclusions', {})
    _excl_labels = {
        'E1_not_llm': 'Not LLM-related',
        'E2_not_gi_onc': 'Not GI oncology',
        'E3_review': 'Review / meta-analysis',
        'E4_no_data': 'No original data',
        'E5_abstract_only': 'Abstract only',
        'E6_duplicate': 'Duplicate report',
        'E7_before_2022': 'Published before 2022',
        'duplicate_or_secondary_reports': 'Duplicate or secondary reports',
        'protocol_b_only_reports': 'Supplementary-only reports',
        'not_llm_or_not_in_scope_or_other_noneligible': 'Not LLM or out of scope',
        'preprint_or_source_restricted_from_protocol_a': 'Non-primary or source-restricted',
        'reclassified_from_core_primary_denominator': 'Reassigned after publication-form review',
    }
    excl_items: list[str] = []
    for k, v in raw_excl.items():
        if isinstance(v, (int, float)) and v > 0:
            excl_items.append(f"{_excl_labels.get(k, k.replace('_', ' ').title())} (n = {int(v)})")
    ft_excl_n = eligibility.get(
        'fulltext_exclusions_total',
        sum(v for v in raw_excl.values() if isinstance(v, (int, float))),
    )
    publication_form_reassigned_n = safe_int(raw_excl.get('reclassified_from_core_primary_denominator'))
    excluded_before_comparison_n = max(ft_excl_n - publication_form_reassigned_n, 0)

    # ── Stage colour scheme: (background, accent / border) ───────────
    _sc = {
        'id':   ('#E8EEF6', '#3C5488'),
        'scr':  ('#FEF6E0', '#E69F00'),
        'elig': ('#FDEDE8', '#E64B35'),
        'inc':  ('#E6F5EE', '#00A087'),
    }

    # ── Canvas & layout constants (no in-figure title — caption in manuscript) ─
    cw, ch = 1240, 880
    sb = 44
    ml, mr, mcx = 100, 570, 335.0
    rl, rr, rcx = 660, 1180, 920.0
    _bands = [
        ('id',   'IDENTIFICATION', 20, 190),
        ('scr',  'SCREENING',      190, 363),
        ('elig', 'ELIGIBILITY',    363, 683),
        ('inc',  'INCLUDED',       683, 870),
    ]

    body: list[str] = []

    # Extra CSS for bold box headers (avoids specificity clash with shared styles)
    body.append(
        f'<style>'
        f'.box-hd{{font:700 14.5px Arial,Helvetica,sans-serif;fill:{palette["text"]};}} '
        f'.box-hd-lg{{font:700 16px Arial,Helvetica,sans-serif;fill:{palette["text"]};}} '
        f'.prisma-note{{font:400 11px Arial,Helvetica,sans-serif;fill:{palette["text"]};}} '
        f'.prisma-sub{{font:600 18px Arial,Helvetica,sans-serif;fill:{palette["text"]};}}'
        f'</style>'
    )

    # ── Background stage bands ───────────────────────────────────────
    for key, _, yt, yb in _bands:
        body.append(rect(sb, yt, cw - sb, yb - yt, fill=_sc[key][0]))

    # ── Left sidebar with rotated stage labels ───────────────────────
    for key, label, yt, yb in _bands:
        body.append(rect(0, yt, sb, yb - yt, fill=_sc[key][1]))
        cy = (yt + yb) / 2
        body.append(
            f'<text x="{sb / 2}" y="{cy}" '
            f'transform="rotate(-90,{sb / 2},{cy})" '
            f'text-anchor="middle" font-size="11" font-weight="700" '
            f'fill="#FFFFFF" font-family="Arial,Helvetica,sans-serif" '
            f'letter-spacing="2">{escape(label)}</text>'
        )

    # ── Thin dividers between bands ──────────────────────────────────
    for _, _, yt, _ in _bands[1:]:
        body.append(line(sb, yt, cw, yt, stroke='#D4D4D4', width=0.5))

    # ═══════════ IDENTIFICATION ══════════════════════════════════════
    b1y, b1h = 40, 125
    body.append(rounded_box(ml, b1y, mr - ml, b1h,
        fill='#FFF', stroke=_sc['id'][1], radius=8, extra=' class="card"'))
    body.append(text_block(mcx, b1y + 26,
        ['Records identified from databases'], anchor='middle', css_class='box-hd'))
    db_lines: list[str] = []
    source_line: list[str] = []
    if identification.get('pubmed'):
        source_line.append(f"PubMed (n = {identification['pubmed']})")
    if identification.get('scopus'):
        source_line.append(f"Scopus (n = {identification['scopus']})")
    if source_line:
        db_lines.append(' · '.join(source_line))
    extra_line: list[str] = []
    if identification.get('other_databases'):
        extra_line.append(f"Other databases (n = {identification['other_databases']})")
    if identification.get('registers'):
        extra_line.append(f"Registers (n = {identification['registers']})")
    if extra_line:
        db_lines.append(' · '.join(extra_line))
    db_lines.append(f"Total identified (N = {identification['total_identified']})")
    body.append(text_block(mcx, b1y + 52, db_lines,
        anchor='middle', css_class='body', line_height=20))

    # Arrow → Screening
    b2y, b2h = 213, 75
    body.append(line(mcx, b1y + b1h + 8, mcx, b2y - 8, stroke=palette['gray'], marker_end=True))
    dup = screening['duplicates_removed']
    if dup:
        body.append(text_block(mcx + 8, b1y + b1h + 14,
            [f"Duplicates removed (n = {dup})"], css_class='prisma-note'))

    # ═══════════ SCREENING ═══════════════════════════════════════════
    body.append(rounded_box(ml, b2y, mr - ml, b2h,
        fill='#FFF', stroke=_sc['scr'][1], radius=8, extra=' class="card"'))
    body.append(text_block(mcx, b2y + 26,
        ['Records screened'], anchor='middle', css_class='box-hd'))
    body.append(text_block(mcx, b2y + 50,
        [f"(n = {screening['records_screened']})"], anchor='middle', css_class='body'))

    # Right: records excluded
    body.append(rounded_box(rl, b2y, rr - rl, b2h,
        fill='#FFF', stroke=_sc['scr'][1], radius=8, extra=' class="card"'))
    body.append(text_block(rcx, b2y + 26,
        ['Records excluded'], anchor='middle', css_class='box-hd'))
    body.append(text_block(rcx, b2y + 50,
        [f"(n = {screening['records_excluded_title_abstract']})"],
        anchor='middle', css_class='body'))
    body.append(line(mr + 8, b2y + b2h / 2, rl - 8, b2y + b2h / 2,
        stroke=palette['gray'], marker_end=True))

    # Arrow → Retrieval
    b3y, b3h = 393, 68
    body.append(line(mcx, b2y + b2h + 8, mcx, b3y - 8, stroke=palette['gray'], marker_end=True))

    # ═══════════ ELIGIBILITY — retrieval ═════════════════════════════
    body.append(rounded_box(ml, b3y, mr - ml, b3h,
        fill='#FFF', stroke=_sc['elig'][1], radius=8, extra=' class="card"'))
    body.append(text_block(mcx, b3y + 24,
        ['Reports sought for retrieval'], anchor='middle', css_class='box-hd'))
    body.append(text_block(mcx, b3y + 48,
        [f"(n = {screening['reports_sought']})"], anchor='middle', css_class='body'))

    # Right: not retrieved
    body.append(rounded_box(rl, b3y, rr - rl, b3h,
        fill='#FFF', stroke=_sc['elig'][1], radius=8, extra=' class="card"'))
    body.append(text_block(rcx, b3y + 24,
        ['Reports not retrieved'], anchor='middle', css_class='box-hd'))
    body.append(text_block(rcx, b3y + 48,
        [f"(n = {screening['reports_not_retrieved']})"], anchor='middle', css_class='body'))
    body.append(line(mr + 8, b3y + b3h / 2, rl - 8, b3y + b3h / 2,
        stroke=palette['gray'], marker_end=True))

    # Arrow → Full-text assessment
    b4y, b4h = 505, 68
    body.append(line(mcx, b3y + b3h + 8, mcx, b4y - 8, stroke=palette['gray'], marker_end=True))

    # ═══════════ ELIGIBILITY — full-text assessment ══════════════════
    body.append(rounded_box(ml, b4y, mr - ml, b4h,
        fill='#FFF', stroke=_sc['elig'][1], radius=8, extra=' class="card"'))
    body.append(text_block(mcx, b4y + 24,
        ['Reports assessed for eligibility'], anchor='middle', css_class='box-hd'))
    body.append(text_block(mcx, b4y + 48,
        [f"(n = {eligibility['reports_assessed_fulltext']})"],
        anchor='middle', css_class='body'))

    # Right: full-text exclusions with itemised reasons
    n_ex = len(excl_items)
    ex_h = max(120, 42 + max(n_ex, 1) * 20 + 48)
    ex_y = max(b4y + b4h / 2 - ex_h / 2, b3y + b3h + 12)
    body.append(rounded_box(rl, ex_y, rr - rl, ex_h,
        fill='#FFF', stroke=_sc['elig'][1], radius=8, extra=' class="card"'))
    body.append(text_block(rcx, ex_y + 22,
        [f"Reports excluded or reassigned (n = {ft_excl_n})"], anchor='middle', css_class='box-hd'))
    if publication_form_reassigned_n > 0:
        summary_lines = [
            f"Excluded before peer-reviewed comparison set finalization (n = {excluded_before_comparison_n})",
            f"Reassigned after publication-form review (n = {publication_form_reassigned_n})",
        ]
        body.append(text_block(rcx, ex_y + 42, summary_lines, anchor='middle', css_class='small', line_height=16))
        body[-1] = body[-1].replace('class="small"', 'class="prisma-note"')
        reason_y = ex_y + 78
    else:
        reason_y = ex_y + 42
    if excl_items:
        body.append(text_block(rcx, reason_y, excl_items,
            anchor='middle', css_class='prisma-note', line_height=20))
    body.append(line(mr + 8, b4y + b4h / 2, rl - 8, b4y + b4h / 2,
        stroke=palette['gray'], marker_end=True))

    # Arrow → Included
    b5y, b5h = 715, 125
    body.append(line(mcx, b4y + b4h + 8, mcx, b5y - 8, stroke=palette['gray'], marker_end=True))

    # ═══════════ INCLUDED ════════════════════════════════════════════
    b5w = rr - ml
    body.append(rounded_box(ml, b5y, b5w, b5h,
        fill='#FFF', stroke=_sc['inc'][1], radius=8, extra=' class="card"'))
    b5cx = ml + b5w / 2
    body.append(text_block(b5cx, b5y + 43,
        ['Included in primary peer-reviewed analysis'], anchor='middle', css_class='box-hd-lg'))
    show_other_methods = any(
        int(other_methods.get(key, 0) or 0) > 0
        for key in ['other_sources', 'reports_sought', 'reports_not_retrieved', 'reports_assessed_fulltext']
    )
    inc_lines = [f"Peer-reviewed full-report studies (n = {included['studies_included_protocol_a']})"]
    if show_other_methods and included.get('studies_included_protocol_b', 0) > 0:
        inc_lines.append(
            f"Supplementary studies summarized separately (n = {included['studies_included_protocol_b']})")
    body.append(text_block(b5cx, b5y + 91, inc_lines,
        anchor='middle', css_class='prisma-sub', line_height=24))

    # ── Assemble SVG ─────────────────────────────────────────────────
    title = 'Figure 1. PRISMA 2020 flow diagram'
    subtitle = 'Study-selection flow for the primary peer-reviewed analysis'
    svg = svg_document(cw, ch, title, subtitle, ''.join(body), palette, max_display_width=1120)
    return title, subtitle, svg


def figure_two_three_combined(records: list[dict[str, Any]], palette: dict[str, str]) -> tuple[str, str, str]:
    """Combined: (A) tier distribution bar chart, (B) CRL × tier, (C) workflow × tier."""

    # ── Shared subsite palette ────────────────────────────────────────
    _clr = {
        'esophageal':  '#E64B35',
        'gastric':     '#00A087',
        'colorectal':  '#3C5488',
        'small_bowel': '#F39B7F',
        'anal':        '#91D1C2',
        'multiple_gi': '#4DBBD5',
        'general_gi':  '#8491B4',
    }

    # ── Data: bar chart ──────────────────────────────────────────────
    counts = {t: Counter(r['gi_subsite'] for r in records if r['tier'] == t) for t in TIERS}
    totals = [sum(counts[t].values()) for t in TIERS]
    y_max = max(totals) if totals else 1
    y_max = int(math.ceil(y_max / 10.0) * 10)
    legend_items = [s for s in GI_SUBSITE_ORDER if any(counts[t].get(s, 0) for t in TIERS)]
    absent_note = tier_s_absent_note(records)
    n_total = len(records)

    # ── Data: heatmaps ───────────────────────────────────────────────
    crl_matrix = [[sum(1 for r in records if r['crl'] == crl and r['tier'] == t) for t in TIERS] for crl in CRL_LEVELS]
    wfs_matrix = [[sum(1 for r in records if r['tier'] == t and stage in r['wfs']) for t in TIERS] for stage in WFS_ORDER]
    crl_max = max((v for row in crl_matrix for v in row), default=1) or 1
    wfs_max = max((v for row in wfs_matrix for v in row), default=1) or 1

    # ── Canvas & global layout ────────────────────────────────────────
    cw, ch = 1160, 760

    # Left panel (A): bar chart
    bw, bar_gap = 56, 26            # bar width, inter-bar gap
    tw = bw + bar_gap
    cl = 64                         # chart left (after Y tick labels)
    ct, cb = 38, 618                # chart top/bottom
    cr_bar = cl + len(TIERS) * tw + bar_gap

    # Vertical divider between panels
    div_x = cl + len(TIERS) * tw + bar_gap + 50

    # Right panel (B, C): heatmaps
    rx0 = div_x + 20               # right panel left margin
    rl_w = 132                      # row label column width
    hx0 = rx0 + rl_w               # heatmap data left edge
    cell_w, cell_gap = 90, 3
    hx1 = hx0 + len(TIERS) * cell_w

    # Right panel Y positions
    crl_cell_h = 52
    wfs_cell_h = 40
    b_label_y = 28
    b_ct = b_label_y + 50          # CRL cells top
    b_cb = b_ct + len(CRL_LEVELS) * crl_cell_h
    b_lg_y = b_cb + 14
    c_label_y = b_lg_y + 40
    c_ct = c_label_y + 50          # WFS cells top
    c_cb = c_ct + len(WFS_ORDER) * wfs_cell_h
    c_lg_y = c_cb + 14

    elements: list[str] = []
    elements.append(
        f'<style>'
        f'.cv-dk{{font:600 12px Arial,Helvetica,sans-serif;fill:{palette["text"]};}} '
        f'.cv-lt{{font:600 12px Arial,Helvetica,sans-serif;fill:#FFFFFF;}} '
        f'.plbl{{font:700 14px Arial,Helvetica,sans-serif;fill:{palette["text"]};}}'
        f'</style>'
    )

    # ════════════ PANEL A: stacked bar chart ═════════════════════════
    # A label: same y as B (b_label_y+14), same x as rotated Y-axis label (cl-36)
    yl_x = cl - 36
    yl_y = (ct + cb) / 2
    panel_label_y = b_label_y + 14   # align with b and c

    elements.append(text_block(yl_x, panel_label_y, ['A'], css_class='plbl'))

    elements.append(rect(cl, ct, cr_bar - cl, cb - ct, fill='#FAFBFC', stroke='#E0E4E8', stroke_width=1))

    raw_step = max(1, y_max / 6)
    tick_step = min((s for s in [1, 2, 5, 10, 15, 20, 25, 50, 100] if s >= raw_step), default=100)
    for tick in range(0, y_max + 1, tick_step):
        y = scale_y(tick, 0, y_max, cb, ct)
        if tick > 0:
            elements.append(line(cl, y, cr_bar, y, stroke='#E0E4E8', width=0.75, dash='3 5'))
        elements.append(text_block(cl - 5, y + 4, [str(tick)], css_class='small', anchor='end'))

    # Rotated Y-axis label
    elements.append(
        f'<text x="{yl_x}" y="{yl_y}" transform="rotate(-90,{yl_x},{yl_y})" '
        f'text-anchor="middle" font-size="12" font-weight="600" '
        f'fill="{palette["text"]}" font-family="Arial,Helvetica,sans-serif">'
        f'Number of studies</text>'
    )

    # Bars — no white stroke so x-axis stays unbroken; axes drawn after bars
    n_leg = len(legend_items)
    for ti, tier in enumerate(TIERS):
        tc = cl + bar_gap + tw * ti + bw / 2
        bl = tc - bw / 2
        running = 0
        for subsite in legend_items:
            val = counts[tier].get(subsite, 0)
            if val == 0:
                continue
            yt = scale_y(running + val, 0, y_max, cb, ct)
            yb_seg = scale_y(running, 0, y_max, cb, ct)
            elements.append(rect(bl, yt, bw, yb_seg - yt, _clr.get(subsite, '#9E9E9E')))
            running += val
        elements.append(text_block(tc, cb + 20, [tier], css_class='axis', anchor='middle'))
        total_tier = totals[ti]
        pct = total_tier / n_total * 100 if n_total else 0
        bar_top_y = scale_y(total_tier, 0, y_max, cb, ct)
        elements.append(text_block(tc, bar_top_y - 13, [str(total_tier)], css_class='small', anchor='middle'))
        elements.append(text_block(tc, bar_top_y - 1, [f'{pct:.1f}%'], css_class='tiny', anchor='middle'))

    # Draw axes AFTER bars so they sit on top and stay continuous
    elements.append(line(cl, ct, cl, cb, stroke=palette['text'], width=1.8))
    elements.append(line(cl, cb, cr_bar, cb, stroke=palette['text'], width=1.8))

    elements.append(text_block((cl + cr_bar) / 2, cb + 40, ['Evidence tier'], css_class='axis', anchor='middle'))

    # Legend — single row, centred within bar chart area (cl → cr_bar)
    bar_area_w = cr_bar - cl
    item_w_leg = int(bar_area_w / n_leg)
    leg_total = n_leg * item_w_leg
    leg_x0 = cl + (bar_area_w - leg_total) / 2
    leg_y0 = 682
    for i, subsite in enumerate(legend_items):
        lx = leg_x0 + i * item_w_leg
        elements.append(circle(lx + 4, leg_y0 + 5, 4, fill=_clr.get(subsite, '#9E9E9E')))
        elements.append(text_block(lx + 11, leg_y0 + 10, [subsite.replace('_', ' ').title()], css_class='tiny'))

    # ════════════ Helper: draw one heatmap ═══════════════════════════
    def _draw_hm(matrix: list[list[int]], row_labels: list[str],
                 ct_hm: float, cell_h: float, fill_color: str, max_val: int) -> None:
        # column headers (TIERS)
        for ci, lab in enumerate(TIERS):
            x = hx0 + ci * cell_w + cell_w / 2
            elements.append(text_block(x, ct_hm - 12, [lab], css_class='axis', anchor='middle'))
        # rows
        for ri, rl in enumerate(row_labels):
            y = ct_hm + ri * cell_h
            lbl = wrap_label(rl.replace('_', ' '), 14)[:2]
            lbl_y = y + cell_h / 2 + (4 if len(lbl) == 1 else -3)
            elements.append(text_block(hx0 - 8, lbl_y, lbl, css_class='axis', anchor='end', line_height=13))
            for ci, val in enumerate(matrix[ri]):
                x = hx0 + ci * cell_w
                w = val / max_val
                fill = blend('#FFFFFF', fill_color, w ** 0.75)
                elements.append(
                    f'<rect x="{x + cell_gap/2}" y="{y + cell_gap/2}" '
                    f'width="{cell_w - cell_gap}" height="{cell_h - cell_gap}" '
                    f'rx="3" fill="{fill}" stroke="#E8ECF0" stroke-width="1" />')
                cls = 'cv-lt' if w > 0.5 else 'cv-dk'
                elements.append(text_block(x + cell_w / 2, y + cell_h / 2 + 4, [str(val)], css_class=cls, anchor='middle'))
        # colour-scale bar
        lg_y = ct_hm + len(row_labels) * cell_h + 12
        lg_w, lg_h, n_steps = 130, 9, 35
        sw = lg_w / n_steps
        for i in range(n_steps):
            t = i / (n_steps - 1)
            elements.append(rect(hx0 + i * sw, lg_y, sw + 0.5, lg_h, blend('#FFFFFF', fill_color, t ** 0.75)))
        elements.append(f'<rect x="{hx0}" y="{lg_y}" width="{lg_w}" height="{lg_h}" rx="1" fill="none" stroke="#CCCCCC" stroke-width="0.5" />')
        elements.append(text_block(hx0, lg_y + lg_h + 11, ['0'], css_class='tiny'))
        elements.append(text_block(hx0 + lg_w, lg_y + lg_h + 11, [str(max_val)], css_class='tiny', anchor='end'))

    # ════════════ PANEL B: CRL heatmap ═══════════════════════════════
    elements.append(text_block(rx0 - 14, b_label_y + 14, ['B'], css_class='plbl'))
    elements.append(text_block(rx0 + 4, b_label_y + 14, ['Clinical risk level × evidence tier'], css_class='axis'))
    _draw_hm(crl_matrix, CRL_LEVELS, b_ct, crl_cell_h, '#3C5488', crl_max)

    # ════════════ PANEL C: Workflow heatmap ══════════════════════════
    elements.append(text_block(rx0 - 14, c_label_y + 14, ['C'], css_class='plbl'))
    elements.append(text_block(rx0 + 4, c_label_y + 14, ['Workflow stage × evidence tier'], css_class='axis'))
    _draw_hm(wfs_matrix, WFS_ORDER, c_ct, wfs_cell_h, '#4DBBD5', wfs_max)

    title = 'Figure 2. Tier distribution, clinical risk level, and workflow composition'
    subtitle = f'(A) study count by tier and GI subsite; (B) CRL × tier; (C) workflow × tier{absent_note}'
    svg = svg_document(cw, ch, title, subtitle, ''.join(elements), palette)
    return title, subtitle, svg


def heatmap_svg(
    matrix: list[list[int]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    subtitle: str,
    palette: dict[str, str],
    fill_color: str,
) -> tuple[str, str, str]:
    """Heatmap — academic journal style, no in-figure title."""
    rows = len(row_labels)
    cols = len(col_labels)
    max_val = max((v for row in matrix for v in row), default=1) or 1

    # ── Dynamic cell sizing ──────────────────────────────────────────
    cell_w, gap = 130, 3
    cell_h = min(56, max(40, 350 // rows))
    cl = 220
    ct = 54
    cr = cl + cols * cell_w
    cb = ct + rows * cell_h
    cw = cr + 60
    ch = cb + 76

    elements: list[str] = []

    # Inject CSS for cell value colours (inline style beats class specificity)
    elements.append(
        f'<style>'
        f'.cv-dk{{font:600 14px Arial,Helvetica,sans-serif;fill:{palette["text"]};}} '
        f'.cv-lt{{font:600 14px Arial,Helvetica,sans-serif;fill:#FFFFFF;}}'
        f'</style>'
    )

    # ── Column headers ───────────────────────────────────────────────
    for ci, lab in enumerate(col_labels):
        x = cl + ci * cell_w + cell_w / 2
        elements.append(text_block(x, ct - 16,
            wrap_label(lab.replace('_', ' '), 14),
            css_class='axis', anchor='middle', line_height=16))

    # ── Row labels + cells ───────────────────────────────────────────
    for ri, row_lab in enumerate(row_labels):
        y = ct + ri * cell_h
        elements.append(text_block(cl - 16, y + cell_h / 2 + 4,
            [row_lab.replace('_', ' ')],
            css_class='axis', anchor='end'))
        for ci, val in enumerate(matrix[ri]):
            x = cl + ci * cell_w
            w = val / max_val
            fill = blend('#FFFFFF', fill_color, w ** 0.75)
            elements.append(
                f'<rect x="{x + gap / 2}" y="{y + gap / 2}" '
                f'width="{cell_w - gap}" height="{cell_h - gap}" '
                f'rx="4" fill="{fill}" stroke="#E8ECF0" stroke-width="1" />')
            cls = 'cv-lt' if w > 0.5 else 'cv-dk'
            elements.append(text_block(
                x + cell_w / 2, y + cell_h / 2 + 5,
                [str(val)], css_class=cls, anchor='middle'))

    # ── Colour-scale legend ──────────────────────────────────────────
    lg_y = cb + 22
    lg_x = cl
    lg_w, lg_h = 200, 12
    n_steps = 50
    sw = lg_w / n_steps
    for i in range(n_steps):
        t = i / (n_steps - 1)
        elements.append(rect(
            lg_x + i * sw, lg_y, sw + 0.5, lg_h,
            blend('#FFFFFF', fill_color, t ** 0.75)))
    elements.append(
        f'<rect x="{lg_x}" y="{lg_y}" width="{lg_w}" height="{lg_h}" '
        f'rx="2" fill="none" stroke="#CCCCCC" stroke-width="0.5" />')
    elements.append(text_block(lg_x, lg_y + lg_h + 14,
        ['0'], css_class='tiny'))
    elements.append(text_block(lg_x + lg_w, lg_y + lg_h + 14,
        [str(max_val)], css_class='tiny', anchor='end'))
    elements.append(text_block(lg_x + lg_w + 14, lg_y + lg_h / 2 + 4,
        ['count'], css_class='tiny'))

    svg = svg_document(cw, ch, title, subtitle, ''.join(elements), palette)
    return title, subtitle, svg


def _figure_three_four_combined_unused(records: list[dict[str, Any]], palette: dict[str, str]) -> tuple[str, str, str]:
    """Superseded by figure_two_three_combined."""
    crl_matrix = [[sum(1 for r in records if r['crl'] == crl and r['tier'] == tier) for tier in TIERS] for crl in CRL_LEVELS]
    wfs_matrix = [[sum(1 for r in records if r['tier'] == tier and stage in r['wfs']) for tier in TIERS] for stage in WFS_ORDER]

    crl_rows, wfs_rows = len(CRL_LEVELS), len(WFS_ORDER)
    cols = len(TIERS)
    crl_max = max((v for row in crl_matrix for v in row), default=1) or 1
    wfs_max = max((v for row in wfs_matrix for v in row), default=1) or 1

    cell_w, gap = 110, 3
    crl_cell_h = 48
    wfs_cell_h = min(48, max(36, 320 // wfs_rows))
    cl = 190

    # Panel A: CRL
    a_label_y = 12
    a_ct = 52
    a_cb = a_ct + crl_rows * crl_cell_h
    # Panel B: WFS
    b_label_y = a_cb + 50
    b_ct = b_label_y + 40
    b_cb = b_ct + wfs_rows * wfs_cell_h

    cr = cl + cols * cell_w
    cw = cr + 60
    ch = b_cb + 76

    elements: list[str] = []
    elements.append(
        f'<style>'
        f'.cv-dk{{font:600 13px Arial,Helvetica,sans-serif;fill:{palette["text"]};}} '
        f'.cv-lt{{font:600 13px Arial,Helvetica,sans-serif;fill:#FFFFFF;}} '
        f'.panel-label{{font:700 13px Arial,Helvetica,sans-serif;fill:{palette["text"]};}}'
        f'</style>'
    )

    def _draw_heatmap(matrix: list[list[int]], row_labels: list[str], col_labels: list[str],
                      ct: float, cell_h: float, fill_color: str, max_val: int, show_col_headers: bool) -> None:
        if show_col_headers:
            for ci, lab in enumerate(col_labels):
                x = cl + ci * cell_w + cell_w / 2
                elements.append(text_block(x, ct - 14,
                    wrap_label(lab.replace('_', ' '), 14),
                    css_class='axis', anchor='middle', line_height=14))
        for ri, row_lab in enumerate(row_labels):
            y = ct + ri * cell_h
            elements.append(text_block(cl - 14, y + cell_h / 2 + 4,
                [row_lab.replace('_', ' ')], css_class='axis', anchor='end'))
            for ci, val in enumerate(matrix[ri]):
                x = cl + ci * cell_w
                w = val / max_val
                fill = blend('#FFFFFF', fill_color, w ** 0.75)
                elements.append(
                    f'<rect x="{x + gap / 2}" y="{y + gap / 2}" '
                    f'width="{cell_w - gap}" height="{cell_h - gap}" '
                    f'rx="4" fill="{fill}" stroke="#E8ECF0" stroke-width="1" />')
                cls = 'cv-lt' if w > 0.5 else 'cv-dk'
                elements.append(text_block(x + cell_w / 2, y + cell_h / 2 + 5,
                    [str(val)], css_class=cls, anchor='middle'))
        # Colour-scale legend bar
        lg_y = ct + len(row_labels) * cell_h + 14
        lg_x, lg_w, lg_h = cl, 160, 10
        n_steps = 40
        sw = lg_w / n_steps
        for i in range(n_steps):
            t = i / (n_steps - 1)
            elements.append(rect(lg_x + i * sw, lg_y, sw + 0.5, lg_h,
                blend('#FFFFFF', fill_color, t ** 0.75)))
        elements.append(
            f'<rect x="{lg_x}" y="{lg_y}" width="{lg_w}" height="{lg_h}" '
            f'rx="2" fill="none" stroke="#CCCCCC" stroke-width="0.5" />')
        elements.append(text_block(lg_x, lg_y + lg_h + 12, ['0'], css_class='tiny'))
        elements.append(text_block(lg_x + lg_w, lg_y + lg_h + 12, [str(max_val)], css_class='tiny', anchor='end'))

    # Panel A label
    elements.append(text_block(14, a_label_y + 14, ['a'], css_class='panel-label'))
    elements.append(text_block(30, a_label_y + 14, ['Clinical risk level × evidence tier'], css_class='axis'))
    _draw_heatmap(crl_matrix, CRL_LEVELS, TIERS, a_ct, crl_cell_h, palette['navy'], crl_max, True)

    # Panel B label
    elements.append(text_block(14, b_label_y + 14, ['b'], css_class='panel-label'))
    elements.append(text_block(30, b_label_y + 14, ['Workflow stage × evidence tier'], css_class='axis'))
    _draw_heatmap(wfs_matrix, WFS_ORDER, TIERS, b_ct, wfs_cell_h, palette['purple'], wfs_max, True)

    absent = tier_s_absent_note(records)
    title = 'Figure 3. Clinical risk and workflow distribution by tier'
    subtitle = f'(A) Clinical risk level and (B) workflow stage concentration across evidence tiers{absent}'
    svg = svg_document(cw, ch, title, subtitle, ''.join(elements), palette)
    return title, subtitle, svg


def figure_five_temporal(input_dir: pathlib.Path, palette: dict[str, str]) -> tuple[str, str, str]:
    """Volume growth and maturity trends — dual panel, academic journal style."""
    yearly = load_yearly_maturity_summary(input_dir)
    maturity = load_json(input_dir / 'statistics' / 'maturity_trends.json')
    years = [row['year'] for row in yearly]
    study_counts = [row['study_count'] for row in yearly]
    cumulative_counts = []
    running_total = 0
    for value in study_counts:
        running_total += value
        cumulative_counts.append(running_total)
    high_evidence = [row['high_evidence_share_percent'] for row in yearly]
    tripod_core = [row['tripod_core_mean_percent'] for row in yearly]
    readiness_higher = [row['readiness_higher_stage_share_percent'] for row in yearly]

    trend_panels = [
        ('Tier maturity rank', maturity['study_level']['year_to_tier_rank'], palette['navy']),
        ('Core-14 completeness', maturity['tripod_core']['year_to_core14_score'], palette['teal']),
        ('Readiness higher-stage rank', maturity['readiness_use_case']['year_to_readiness_rank'], palette['orange']),
    ]
    maturity_series = [
        ('High-evidence share', high_evidence, palette['navy']),
        ('TRIPOD core-14 mean %', tripod_core, palette['teal']),
        ('Higher-stage readiness share', readiness_higher, palette['orange']),
    ]

    count_y_max = int(math.ceil(max(max(study_counts, default=1), max(cumulative_counts, default=1)) / 10.0) * 10)
    count_y_max = max(count_y_max, 20)
    pct_y_max = 100

    n_years = len(years)
    data_w = max(300, min(700, n_years * 140))
    vl = 60                           # tight left margin: tick labels end at vl+28=88
    axis_x = vl + 46                  # Y-axis line position
    vr = vl + 46 + data_w + 30       # right edge of chart area
    cw = vr + 36                      # canvas: small right margin
    ch = 920
    vt, vb = 76, 340
    ml, mr = vl, vr
    mt, mb = 468, 700
    # left padding so first year doesn't sit on the Y-axis
    year_xv = {yr: scale_x(yr, years[0] - 0.5, years[-1] + 0.3, axis_x, vr - 16) for yr in years}
    year_xm = {yr: scale_x(yr, years[0] - 0.5, years[-1] + 0.3, axis_x, mr - 16) for yr in years}

    elements = [
        text_block(vl - 26, 24, ['A'], css_class='axis', size=14, weight=700),
        text_block(vl, 24, ['Volume growth'], css_class='axis'),
        text_block(vl, 48, ['Bars = annual study count; line = cumulative included studies'], css_class='small'),
        text_block(ml - 26, 418, ['B'], css_class='axis', size=14, weight=700),
        text_block(ml, 418, ['Maturity growth (%)'], css_class='axis'),
        text_block(ml, 442, ['High-evidence share, TRIPOD core-14 completeness, and higher-stage readiness share'], css_class='small'),
    ]

    # --- Volume panel ---
    for tick in range(0, count_y_max + 1, max(10, count_y_max // 5 or 1)):
        y = scale_y(tick, 0, count_y_max, vb, vt)
        elements.append(line(axis_x, y, vr - 16, y, stroke=palette['grid'], width=1, dash='4 6'))
        elements.append(text_block(axis_x - 8, y + 5, [str(tick)], css_class='small', anchor='end'))
    elements.append(line(axis_x, vt, axis_x, vb, stroke=palette['text'], width=1.6))
    elements.append(line(axis_x, vb, vr - 16, vb, stroke=palette['text'], width=1.8))

    bar_width = min(data_w / max(n_years, 1) * 0.42, 56)
    for year, count, cumulative in zip(years, study_counts, cumulative_counts):
        xc = year_xv[year]
        xb = xc - bar_width / 2
        yb = scale_y(count, 0, count_y_max, vb, vt)
        elements.append(rect(xb, yb, bar_width, vb - yb, blend(palette['navy'], '#ffffff', 0.22)))
        elements.append(text_block(xc, yb - 10, [str(count)], css_class='small', anchor='middle'))
        elements.append(text_block(xc, vb + 28, [str(year)], css_class='axis', anchor='middle'))
        cy = scale_y(cumulative, 0, count_y_max, vb, vt)
        elements.append(circle(xc, cy, 5.6, palette['orange'], stroke='#ffffff', stroke_width=2))
    cum_pts = [(year_xv[yr], scale_y(v, 0, count_y_max, vb, vt)) for yr, v in zip(years, cumulative_counts)]
    elements.append(polyline(cum_pts, stroke=palette['orange'], width=4))
    # Inline volume legend
    elements.append(rect(axis_x + 14, vt + 6, 13, 13, blend(palette['navy'], '#ffffff', 0.22)))
    elements.append(text_block(axis_x + 32, vt + 17, ['Annual count'], css_class='tiny'))
    elements.append(line(axis_x + 120, vt + 12, axis_x + 144, vt + 12, stroke=palette['orange'], width=3))
    elements.append(circle(axis_x + 132, vt + 12, 4, palette['orange'], stroke='#ffffff', stroke_width=1.5))
    elements.append(text_block(axis_x + 150, vt + 17, ['Cumulative'], css_class='tiny'))

    # --- Maturity panel ---
    for tick in range(0, pct_y_max + 1, 20):
        y = scale_y(tick, 0, pct_y_max, mb, mt)
        elements.append(line(axis_x, y, mr - 16, y, stroke=palette['grid'], width=1, dash='4 6'))
        elements.append(text_block(axis_x - 8, y + 5, [str(tick)], css_class='small', anchor='end'))
    elements.append(line(axis_x, mt, axis_x, mb, stroke=palette['text'], width=1.6))
    elements.append(line(axis_x, mb, mr - 16, mb, stroke=palette['text'], width=1.8))

    for label, values, color in maturity_series:
        points = [(year_xm[yr], scale_y(v, 0, pct_y_max, mb, mt)) for yr, v in zip(years, values)]
        elements.append(polyline(points, stroke=color, width=3.5))
        for x, y in points:
            elements.append(circle(x, y, 4.8, color, stroke='#ffffff', stroke_width=1.8))
        lx, ly = points[-1]
        if ly >= mb - 10:
            label_y = ly - 10
        elif ly <= mt + 10:
            label_y = ly + 14
        else:
            label_y = ly + 5
        elements.append(text_block(lx + 12, label_y, [pct_label(values[-1], 1)], css_class='tiny'))
    for year in years:
        elements.append(text_block(year_xm[year], mb + 28, [str(year)], css_class='axis', anchor='middle'))

    # Centered single-row legend — centered relative to full canvas width
    chart_left = axis_x
    chart_right = vr - 16
    chart_span = chart_right - chart_left
    n_leg = len(maturity_series)
    item_w = min(180, chart_span // n_leg)
    lx0 = (cw - n_leg * item_w) / 2
    leg_y = 762
    for idx, (label, _, color) in enumerate(maturity_series):
        x = lx0 + idx * item_w
        elements.append(line(x, leg_y, x + 26, leg_y, stroke=color, width=4))
        elements.append(circle(x + 13, leg_y, 5, color, stroke='#ffffff', stroke_width=2))
        elements.append(text_block(x + 34, leg_y + 5, [label], css_class='small'))

    # Trend statistics — evenly spaced, centered relative to full canvas width
    note_y = 796
    n_panels = len(trend_panels)
    stat_slot = cw / n_panels          # divide full canvas into equal thirds
    for idx, (label, stats, color) in enumerate(trend_panels):
        slot_cx = stat_slot * (idx + 0.5)   # centre of each canvas slice
        elements.append(text_block(slot_cx, note_y + 22, [label], css_class='small', anchor='middle'))
        elements.append(line(slot_cx - 28, note_y + 36, slot_cx + 28, note_y + 36, stroke=color, width=3))
        elements.append(text_block(slot_cx, note_y + 56, [f"ρ={safe_float_num(stats.get('rho')):.3f}  p={safe_float_num(stats.get('p_value_normal_approx')):.4f}"], css_class='axis', anchor='middle'))

    title = 'Figure 3. Volume growth and maturity trends'
    subtitle = '(A) annual study growth; (B) descriptive maturity trends'
    svg = svg_document(cw, ch, title, subtitle, ''.join(elements), palette)
    return title, subtitle, svg


def figure_six_tripod(input_dir: pathlib.Path, palette: dict[str, str]) -> tuple[str, str, str]:
    summary = load_json(input_dir / 'statistics' / 'tripod_dual_track_summary.json')
    rows = sorted(
        load_csv(input_dir / 'statistics' / 'tripod_item_grouping.csv'),
        key=lambda row: (0 if row['item_group'] == 'core' else 1, -safe_float_num(row['reporting_rate_percent'])),
    )
    core_rows = [row for row in rows if row['item_group'] == 'core']
    extended_rows = [row for row in rows if row['item_group'] == 'extended']
    core_mean = safe_float_num(summary['core_14_item_operational']['mean_percent'])
    full_mean = safe_float_num(summary['full_19_item']['mean_percent'])
    extended_mean = safe_float_num(summary['item_group_summary']['extended_mean_reporting_rate_percent'])
    scored = safe_int(summary.get('scored_records'))
    total = safe_int(summary.get('total_included_records'))
    coverage_pct = (scored / total * 100.0) if total else 0.0

    width = 1120
    elements: list[str] = []

    # ── Shared panel drawing helper ──────────────────────────────────
    def draw_group_panel(
        panel_x: float,
        panel_y: float,
        panel_w: float,
        n_rows: int,
        row_h: float,
        *,
        label: str,
        group_rows: list[dict[str, str]],
        fill_color: str,
        stroke_color: str,
        mean_pct: float,
    ) -> float:
        """Draw panel and return bottom y position."""
        elements.append(line(panel_x, panel_y, panel_x + panel_w, panel_y, stroke=stroke_color, width=2))
        elements.append(text_block(panel_x + 16, panel_y + 22, [label], css_class='axis'))
        elements.append(text_block(panel_x + panel_w - 10, panel_y + 22, [f'Mean {mean_pct:.1f}%'], css_class='axis', anchor='end'))

        chart_left = panel_x + 280
        chart_right = panel_x + panel_w - 16
        chart_top = panel_y + 40
        chart_bottom = chart_top + n_rows * row_h
        row_slot = row_h

        for tick in range(0, 101, 20):
            x = scale_x(tick, 0, 100, chart_left, chart_right)
            elements.append(line(x, chart_top, x, chart_bottom, stroke=palette['grid'], width=1, dash='4 6'))
            elements.append(text_block(x, chart_bottom + 16, [str(tick)], css_class='tiny', anchor='middle'))
        elements.append(line(chart_left, chart_top, chart_left, chart_bottom, stroke=palette['text'], width=1.4))
        elements.append(line(chart_left, chart_bottom, chart_right, chart_bottom, stroke=palette['text'], width=1.4))

        for index, row in enumerate(group_rows):
            slot_top = chart_top + index * row_slot
            value = safe_float_num(row.get('reporting_rate_percent'))
            bar_h = max(12.0, row_slot - 6)
            bar_y = slot_top + (row_slot - bar_h) / 2
            bar_w = scale_x(value, 0, 100, chart_left, chart_right) - chart_left
            label_text = TRIPOD_ITEM_LABELS.get(str(row.get('tripod_item')), str(row.get('tripod_item', '')).replace('_', ' '))
            # No gray background rect — only the coloured data bar
            elements.append(rect(chart_left, bar_y, max(bar_w, 0), bar_h, blend(fill_color, '#ffffff', 0.08)))
            elements.append(text_block(chart_left - 10, bar_y + bar_h / 2 + 4, [label_text], css_class='tiny', anchor='end'))
            elements.append(text_block(chart_left + max(bar_w, 0) + 6, bar_y + bar_h / 2 + 4, [fmt_pct(value)], css_class='tiny'))

        # Mean dashed line — drawn AFTER bars so it sits on top
        mean_x = scale_x(mean_pct, 0, 100, chart_left, chart_right)
        elements.append(line(mean_x, chart_top, mean_x, chart_bottom, stroke=stroke_color, width=1.0, dash='5 4', opacity=0.4))

        return chart_bottom + 28  # bottom including tick labels

    # ── Core panel (14 items) ────────────────────────────────────────
    core_bottom = draw_group_panel(
        40, 96, 1040, len(core_rows), 24,
        label='Core universal items (14)',
        group_rows=core_rows,
        fill_color=palette['navy'],
        stroke_color=palette['navy'],
        mean_pct=core_mean,
    )

    # ── Extended panel (5 items) ─────────────────────────────────────
    ext_bottom = draw_group_panel(
        40, core_bottom + 10, 1040, len(extended_rows), 28,
        label='Extended transparency items (5)',
        group_rows=extended_rows,
        fill_color=palette['orange'],
        stroke_color=palette['orange'],
        mean_pct=extended_mean,
    )

    height = int(ext_bottom + 12)
    title = 'Figure 4. TRIPOD-LLM reporting profile'
    subtitle = 'Core-14 versus full-19 reporting completeness and item-level rates'
    svg = svg_document(width, height, title, subtitle, ''.join(elements), palette, max_display_width=1080)
    return title, subtitle, svg


def figure_seven_readiness(input_dir: pathlib.Path, palette: dict[str, str]) -> tuple[str, str, str]:
    readiness = load_json(input_dir / 'readiness' / 'clinical_readiness_report.json')
    counts = readiness['readiness_counts']
    overlay = readiness['overlay_tag_counts']
    use_case_count = safe_int(readiness.get('use_case_count'))
    record_count = safe_int(readiness.get('record_count'))
    validation = readiness.get('validation_sampling', {})
    stage_rows = [
        ('not_ready', 'Not ready for clinical use', '#E64B35', ''),
        ('external_validation_needed', 'External-validation candidates', '#F39B7F', ''),
        ('human_review_only', 'Low-risk, human-review-only', '#4DBBD5', ''),
        ('prospective_trial_candidate', 'Prospective-trial candidates', '#00A087', ''),
    ]
    overlay_rows = [
        ('needs_real_world_validation', 'Needs real-world validation', '#3C5488', ''),
        ('needs_external_multicenter_validation', 'Needs external / multicenter', '#F39B7F', ''),
        ('needs_safety_audit', 'Needs safety review', '#8491B4', ''),
        ('rct_priority', 'Randomized-trial priority', '#00A087', ''),
        ('patient_facing_high_risk', 'Patient-facing high risk', '#E64B35', ''),
    ]
    width = 1060
    elements: list[str] = []

    # ── Compact summary strip ────────────────────────────────────────
    sl, sr = 40, 1020
    st, sb = 14, 76
    # ── Shared helper: horizontal bar section ────────────────────────
    def draw_bar_section(
        section_y: float,
        section_label: str,
        bar_rows: list[tuple[str, str, str, str]],
        data_source: dict,
        chart_left: float,
        accent: str,
    ) -> float:
        """Draw a labelled horizontal bar section. Returns bottom y."""
        chart_right = 1000
        elements.append(line(sl, section_y, sr, section_y, stroke=accent, width=1.5))
        elements.append(text_block(sl, section_y + 20, [section_label], css_class='axis'))

        n = len(bar_rows)
        row_h = 36
        chart_top = section_y + 34
        chart_bottom = chart_top + n * row_h

        max_val = max((safe_int(data_source.get(key)) for key, _, _, _ in bar_rows), default=1)
        max_val = int(math.ceil(max_val / 25.0) * 25) or 25
        tick_step = max(25, max_val // 4 or 25)
        for tick in range(0, max_val + 1, tick_step):
            x = scale_x(tick, 0, max_val, chart_left, chart_right)
            elements.append(line(x, chart_top, x, chart_bottom, stroke=palette['grid'], width=1, dash='4 6'))
            elements.append(text_block(x, chart_bottom + 14, [str(tick)], css_class='tiny', anchor='middle'))
        elements.append(line(chart_left, chart_top, chart_left, chart_bottom, stroke=palette['text'], width=1.4))
        elements.append(line(chart_left, chart_bottom, chart_right, chart_bottom, stroke=palette['text'], width=1.4))

        for i, (key, label, color, _) in enumerate(bar_rows):
            y = chart_top + i * row_h
            value = safe_int(data_source.get(key))
            pct = (value / use_case_count * 100.0) if use_case_count else 0.0
            bar_h = row_h - 8
            bar_y = y + 4
            bar_w = scale_x(value, 0, max_val, chart_left, chart_right) - chart_left
            elements.append(rect(chart_left, bar_y, max(bar_w, 0), bar_h, color))
            # Label left of chart — single line only, no descriptor
            elements.append(text_block(chart_left - 8, bar_y + bar_h / 2 + 4, [label], css_class='small', anchor='end'))
            elements.append(text_block(chart_left + max(bar_w, 0) + 6, bar_y + bar_h / 2 + 4, [f'{value} ({pct:.0f}%)'], css_class='tiny'))

        return chart_bottom + 26

    # ── Stage counts section ─────────────────────────────────────────
    stage_bottom = draw_bar_section(
        section_y=sb + 20,
        section_label=f'Readiness stage counts (n = {use_case_count})',
        bar_rows=stage_rows,
        data_source=counts,
        chart_left=380,
        accent='#3C5488',
    )

    # ── Overlay section ──────────────────────────────────────────────
    overlay_bottom = draw_bar_section(
        section_y=stage_bottom + 8,
        section_label='Validation, safety, and prioritization overlays',
        bar_rows=overlay_rows,
        data_source=overlay,
        chart_left=380,
        accent='#8491B4',
    )

    height = int(overlay_bottom + 10)
    title = 'Figure 5. Clinical readiness roadmap'
    subtitle = f'Use-case-level readiness stages (n={use_case_count}) and overlay-based trial priorities'
    svg = svg_document(width, height, title, subtitle, ''.join(elements), palette)
    return title, subtitle, svg


def figure_eight_forest(input_dir: pathlib.Path, palette: dict[str, str]) -> tuple[str, str, str]:
    meta_rows = load_csv(input_dir / 'meta' / 'meta_study_table.csv')
    family_results = load_json(input_dir / 'meta' / 'family_results.json')
    rows_by_family_all: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in meta_rows:
        rows_by_family_all[row.get('metric_family', 'other')].append(row)
    displayed_families = [
        family for family in ['accuracy', 'sensitivity']
        if len(rows_by_family_all.get(family, [])) >= 2
    ]
    if not displayed_families:
        displayed_families = [family for family in ['accuracy', 'sensitivity'] if rows_by_family_all.get(family)]
    rows_by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in meta_rows:
        if row['metric_family'] in displayed_families:
            rows_by_family[row['metric_family']].append(row)

    summary_candidates: list[float] = []
    for family in displayed_families:
        for row in rows_by_family[family]:
            effect = safe_float_num(row['effect_size'])
            se = safe_float_num(row['se'])
            summary_candidates.extend([effect, max(0.0, effect - 1.96 * se), min(1.0, effect + 1.96 * se)])
    domain_min = min(0.3, math.floor(min(summary_candidates or [0.3]) * 10) / 10)
    domain_max = max(1.0, math.ceil(max(summary_candidates or [1.0]) * 10) / 10)

    plot_left, plot_right = 440, 900
    estimate_x = 1060
    top = 20
    row_height = 28
    row_y = top + 76
    alt_row_fill = '#F7F9FC'
    study_color = palette['navy']
    family_colors = {'accuracy': palette['navy'], 'sensitivity': palette['teal'], 'specificity': palette['purple'], 'auc_auroc': palette['orange'], 'f1': '#4DBBD5'}

    # ── First pass: collect row positions for alternating backgrounds ──
    row_entries: list[dict[str, Any]] = []
    scan_y = row_y
    for family in displayed_families:
        scan_y += row_height  # family header row
        for row in sorted(rows_by_family[family], key=lambda item: safe_float_num(item['effect_size']), reverse=True):
            row_entries.append({'y': scan_y, 'type': 'study'})
            scan_y += row_height
        scan_y += 18

    content: list[str] = []

    # ── Alternating row backgrounds (drawn first, behind everything) ──
    global_idx = 0
    for entry in row_entries:
        if global_idx % 2 == 1:
            band_h = row_height + 16 if entry['type'] == 'pooled' else row_height
            content.append(rect(30, entry['y'] - row_height / 2 - 2, 1040, band_h, alt_row_fill))
        global_idx += 1

    # ── Column headers ───────────────────────────────────────────────
    content.append(text_block(40, top + 18, ['Study or family'], css_class='axis'))
    content.append(text_block((plot_left + plot_right) / 2, top + 18, ['Effect size and 95% CI'], css_class='axis', anchor='middle'))
    content.append(text_block(estimate_x, top + 18, ['Estimate [95% CI]'], css_class='axis', anchor='end'))
    content.append(line(30, top + 28, 1070, top + 28, stroke=palette['text'], width=1.5))

    # ── Tick axis ────────────────────────────────────────────────────
    axis_y = top + 34
    for tick_index in range(int(domain_min * 10), int(domain_max * 10) + 1):
        tick = tick_index / 10
        x = scale_x(tick, domain_min, domain_max, plot_left, plot_right)
        content.append(line(x, axis_y, x, axis_y + 8, stroke=palette['text'], width=1.5))
        content.append(text_block(x, axis_y + 28, [f'{tick:.1f}'], css_class='small', anchor='middle'))

    for family in displayed_families:
        display_family = family_label(family)
        fam_color = family_colors.get(family, palette['navy'])
        family_payload = family_results.get('results', {}).get(family) or {}
        heterogeneity_text = heterogeneity_note(
            family_payload.get('heterogeneity_result') or family_payload.get('pooled_result')
        )

        # Family header — bold, with a subtle colored left accent
        content.append(line(34, row_y - 4, 34, row_y + 10, stroke=fam_color, width=3))
        content.append(text_block(44, row_y + 4, [f'{display_family} (n = {len(rows_by_family[family])})'], css_class='axis'))
        if heterogeneity_text:
            content.append(text_block(estimate_x, row_y + 4, [heterogeneity_text], css_class='small', anchor='end'))
        row_y += row_height

        for row in sorted(rows_by_family[family], key=lambda item: safe_float_num(item['effect_size']), reverse=True):
            effect = safe_float_num(row['effect_size'])
            se = safe_float_num(row['se'])
            low = max(0.0, effect - 1.96 * se)
            high = min(1.0, effect + 1.96 * se)
            x_low = scale_x(low, domain_min, domain_max, plot_left, plot_right)
            x_high = scale_x(high, domain_min, domain_max, plot_left, plot_right)
            x_mid = scale_x(effect, domain_min, domain_max, plot_left, plot_right)
            box_size = max(5, min(11, math.sqrt(max(safe_int(row.get('sample_size'), 100), 25)) / 2.2))

            content.append(text_block(44, row_y + 4, [forest_study_label(row)], css_class='body'))
            # CI line and whisker caps — uniform color
            content.append(line(x_low, row_y, x_high, row_y, stroke=study_color, width=1.8))
            content.append(line(x_low, row_y - 6, x_low, row_y + 6, stroke=study_color, width=1.5))
            content.append(line(x_high, row_y - 6, x_high, row_y + 6, stroke=study_color, width=1.5))
            # Study square — uniform navy
            content.append(rect(x_mid - box_size / 2, row_y - box_size / 2, box_size, box_size,
                                study_color, stroke='#ffffff', stroke_width=1.0))
            content.append(text_block(estimate_x, row_y + 4, [fmt_ci(effect, low, high)], css_class='small', anchor='end'))
            row_y += row_height
        row_y += 6
        content.append(line(30, row_y, 1070, row_y, stroke=palette['grid'], width=0.75))
        row_y += 18

    # ── Bottom rule ──────────────────────────────────────────────────
    content.append(line(30, row_y - 12, 1070, row_y - 12, stroke=palette['text'], width=1.5))

    height = max(600, int(row_y + 10))
    title = 'Figure 6. Study-level performance ranges across reported metric families'
    subtitle = 'Accuracy-like study-level forest; single-study non-accuracy endpoints are summarized textually'
    svg = svg_document(1100, height, title, subtitle, ''.join(content), palette)
    return title, subtitle, svg


def figure_s1_bayesian_and_ecosystem(input_dir: pathlib.Path, records: list[dict[str, Any]], palette: dict[str, str]) -> tuple[str, str, str]:
    """Figure S1: LLM model ecosystem."""
    width = 1080
    elements: list[str] = []

    counts = Counter(ecosystem_model_family_label(record['llm_model']) for record in records)
    top_models = counts.most_common(8)
    source_counts = Counter(classify_model(record['llm_model']) for record in records)

    elements.append(text_block(40, 18, ['LLM model ecosystem'], css_class='axis'))

    bar_left = 140
    bar_top = 52
    bar_right = 560
    bar_bottom = bar_top + 282
    row_h = (bar_bottom - bar_top) / max(len(top_models), 1)
    max_count = max((value for _, value in top_models), default=1)
    for tick in range(0, max_count + 1, max(5, math.ceil(max_count / 5) or 1)):
        x = scale_x(tick, 0, max_count, bar_left, bar_right)
        elements.append(line(x, bar_top, x, bar_bottom, stroke=palette['grid'], width=1, dash='4 6'))
        elements.append(text_block(x, bar_bottom + 24, [str(tick)], css_class='small', anchor='middle'))
    elements.append(line(bar_left, bar_bottom, bar_right, bar_bottom, stroke=palette['text'], width=1.6))
    elements.append(text_block((bar_left + bar_right) / 2, bar_bottom + 52, ['Study count'], css_class='axis', anchor='middle'))

    bar_palette = ['#3C5488', '#00A087', '#E64B35', '#4DBBD5', '#F39B7F', '#8491B4', '#91D1C2', '#DC9A6C']
    for idx, (model, value) in enumerate(reversed(top_models)):
        y = bar_top + idx * row_h + 5
        label = short_label(model, 22)
        color = bar_palette[len(top_models) - 1 - idx] if (len(top_models) - 1 - idx) < len(bar_palette) else palette['gray']
        bar_w = scale_x(value, 0, max_count, bar_left, bar_right) - bar_left
        elements.append(rect(bar_left, y, bar_w, row_h - 10, color))
        elements.append(text_block(bar_left - 10, y + row_h / 2 + 2, [label], css_class='label', anchor='end'))
        elements.append(text_block(bar_left + bar_w + 8, y + row_h / 2 + 2, [str(value)], css_class='small'))

    donut_cx, donut_cy, donut_r = 830, bar_top + 122, 102
    donut_w = 42
    donut_colors = ['#3C5488', '#00A087', '#B0B8C8']
    donut_labels = [('closed_source', 'Closed source'), ('open_source', 'Open source'), ('other', 'Other / not reported')]
    total = sum(source_counts.values()) or 1
    start_angle = -90.0
    for index, (key, _) in enumerate(donut_labels):
        value = source_counts.get(key, 0)
        sweep = value / total * 360.0
        end_angle = start_angle + sweep
        large_arc = 1 if sweep > 180 else 0
        sx = donut_cx + donut_r * math.cos(math.radians(start_angle))
        sy = donut_cy + donut_r * math.sin(math.radians(start_angle))
        ex = donut_cx + donut_r * math.cos(math.radians(end_angle))
        ey = donut_cy + donut_r * math.sin(math.radians(end_angle))
        elements.append(f'<path d="M {sx:.2f},{sy:.2f} A {donut_r},{donut_r} 0 {large_arc} 1 {ex:.2f},{ey:.2f}" fill="none" stroke="{donut_colors[index]}" stroke-width="{donut_w}" stroke-linecap="butt" />')
        start_angle = end_angle
    elements.append(circle(donut_cx, donut_cy, donut_r - donut_w / 2, '#ffffff'))
    elements.append(text_block(donut_cx, donut_cy - 10, ['Open vs', 'closed-source mix'], css_class='axis', anchor='middle', line_height=16))
    leg_y = donut_cy + donut_r + 42
    leg_x = donut_cx - 106
    legend_step = 30
    for index, (key, label) in enumerate(donut_labels):
        ly = leg_y + index * legend_step
        elements.append(rect(leg_x, ly, 14, 14, donut_colors[index]))
        elements.append(text_block(leg_x + 22, ly + 11, [f'{label} (n={source_counts.get(key, 0)})'], css_class='small', line_height=13))

    height = int(max(bar_bottom + 60, leg_y + legend_step * len(donut_labels) + 18) + 28)
    title = 'Figure S1. LLM model ecosystem'
    subtitle = 'Model-family frequency and source openness across the peer-reviewed study set'
    svg = svg_document(width, height, title, subtitle, ''.join(elements), palette)
    return title, subtitle, svg


def figure_s3_doi(input_dir: pathlib.Path, palette: dict[str, str]) -> tuple[str, str, str]:
    rows = load_csv(input_dir / 'meta' / 'meta_input_audited.csv')
    width, height = 1060, 700
    chart_left, chart_top, chart_right, chart_bottom = 100, 40, 980, 560
    effects = [float(row['effect_size']) for row in rows]
    precisions = [1.0 / max(float(row['se']), 1e-6) for row in rows]
    x_min = math.floor(min(effects) * 10) / 10
    x_max = math.ceil(max(effects) * 10) / 10
    min_precision = max(min(precisions), 1e-6)
    max_precision = max(precisions)
    tick_candidates: list[float] = []
    for exponent in range(math.floor(math.log10(min_precision)) - 1, math.ceil(math.log10(max_precision)) + 2):
        for multiplier in (1, 2, 5):
            tick_candidates.append(multiplier * (10 ** exponent))
    tick_candidates = sorted({tick for tick in tick_candidates if tick > 0})
    first_tick = max((tick for tick in tick_candidates if tick <= min_precision), default=tick_candidates[0])
    last_tick = min((tick for tick in tick_candidates if tick >= max_precision), default=tick_candidates[-1])
    y_ticks = [tick for tick in tick_candidates if first_tick <= tick <= last_tick]
    y_min_log = math.log10(first_tick)
    y_max_log = math.log10(last_tick)
    elements: list[str] = []
    for tick_index in range(int(x_min * 10), int(x_max * 10) + 1):
        tick = tick_index / 10
        x = scale_x(tick, x_min, x_max, chart_left, chart_right)
        elements.append(line(x, chart_top, x, chart_bottom, stroke=palette['grid'], width=1, dash='4 6'))
        elements.append(text_block(x, chart_bottom + 30, [f'{tick:.1f}'], css_class='small', anchor='middle'))
    for tick in y_ticks:
        y = scale_y(math.log10(tick), y_min_log, y_max_log, chart_bottom, chart_top)
        elements.append(line(chart_left, y, chart_right, y, stroke=palette['grid'], width=1, dash='4 6'))
        elements.append(text_block(chart_left - 16, y + 5, [f'{tick:g}'], css_class='small', anchor='end'))
    elements.append(line(chart_left, chart_top, chart_left, chart_bottom, stroke=palette['text'], width=2))
    elements.append(line(chart_left, chart_bottom, chart_right, chart_bottom, stroke=palette['text'], width=2))
    y_label_x = chart_left - 62
    y_label_y = (chart_top + chart_bottom) / 2
    elements.append(
        f'<text x="{y_label_x}" y="{y_label_y}" transform="rotate(-90,{y_label_x},{y_label_y})" '
        f'text-anchor="middle" font-size="12" font-weight="600" '
        f'fill="{palette["text"]}" font-family="Arial,Helvetica,sans-serif">'
        f'Precision (log scale)</text>'
    )
    for row in rows:
        effect = float(row['effect_size'])
        precision = 1.0 / max(float(row['se']), 1e-6)
        x = scale_x(effect, x_min, x_max, chart_left, chart_right)
        y = scale_y(math.log10(precision), y_min_log, y_max_log, chart_bottom, chart_top)
        color = SUBSITE_COLORS.get(row['gi_subsite'], palette['gray'])
        radius = max(4.5, min(8.5, math.sqrt(max(safe_int(row.get('sample_size'), 100), 25)) / 2.2))
        elements.append(circle(x, y, radius, color, stroke='#ffffff', stroke_width=1.4))
    legend_items = [subsite for subsite in GI_SUBSITE_ORDER if any(row['gi_subsite'] == subsite for row in rows)]
    n_leg = len(legend_items)
    item_w = 160
    leg_x0 = (width - n_leg * item_w) / 2
    for index, subsite in enumerate(legend_items):
        x = leg_x0 + index * item_w
        y = 602
        elements.append(rect(x, y, 16, 16, SUBSITE_COLORS.get(subsite, palette['gray'])))
        elements.append(text_block(x + 24, y + 13, [subsite.replace('_', ' ')], css_class='small'))
    title = 'Figure S2. Effect size versus precision plot'
    subtitle = 'Effect size versus precision for the analyzable performance studies (log-scale precision axis)'
    svg = svg_document(width, height, title, subtitle, ''.join(elements), palette)
    return title, subtitle, svg


def figure_s4_publication_form_sensitivity(input_dir: pathlib.Path, palette: dict[str, str]) -> tuple[str, str, str]:
    summary = load_json(input_dir / 'publication_form' / 'publication_form_sensitivity_summary.json')
    form_summary = load_json(input_dir / 'publication_form' / 'publication_form_summary.json')
    tier_distribution = summary.get('tier_distribution', {})
    readiness_dist = summary.get('readiness_summary', {}).get('readiness_stage_distribution', {})
    tripod_summary = summary.get('tripod_summary', {})

    def humanize_token(value: str) -> str:
        return value.replace('_', ' ').replace('-', ' ').strip().capitalize()

    def card(x: float, y: float, w: float, h: float, label: str, value: str, note: str, accent: str, fill: str) -> str:
        return ''.join(
            [
                rounded_box(x, y, w, h, fill=fill, stroke=blend(accent, '#D6DCE5', 0.35), radius=18, extra=' class="card"'),
                text_block(x + 16, y + 23, [label], css_class='small'),
                text_block(x + 16, y + 48, [value], css_class='axis', size=21),
                text_block(x + 16, y + 65, wrap_label(note, 22)[:2], css_class='tiny', line_height=12),
            ]
        )

    def dumbbell_panel(
        x: float,
        y: float,
        w: float,
        h: float,
        panel_label: str,
        title_text: str,
        subtitle_text: str,
        rows: list[dict[str, float | str]],
        scale_max: float,
        axis_label: str,
        denominator_note: str,
    ) -> str:
        max_label_len = max((len(str(row.get('label', ''))) for row in rows), default=4)
        retained_colors = []
        for row in rows:
            color = str(row.get('color'))
            if color not in retained_colors:
                retained_colors.append(color)
        plot_left = x + max(50, min(120, max_label_len * 10 + 16))
        plot_right = x + w - 60
        plot_top = y + 112
        plot_bottom = y + h - 54
        row_h = (plot_bottom - plot_top) / max(len(rows), 1)
        bar_h = 7  # height of each bar
        gap = 3    # vertical gap between Full and Retained bars
        full_color = '#C8CDD5'  # uniform light gray for Full
        parts = [
            rounded_box(x, y, w, h, fill='#FFFFFF', stroke=blend(palette['grid'], '#7A8797', 0.30), radius=20, extra=' class="card"'),
            text_block(x + 18, y + 38, [panel_label], css_class='axis', size=28, weight=700),
            text_block(x + 56, y + 30, [title_text], css_class='axis'),
            text_block(x + 56, y + 50, wrap_label(subtitle_text, 34)[:2], css_class='small', line_height=13),
            rect(x + w - 134, y + 66, 14, bar_h, full_color),
            text_block(x + w - 114, y + 76, ['Full'], css_class='tiny'),
            text_block(x + w - 14, plot_top - 10, ['Δ'], css_class='tiny', anchor='end'),
        ]
        retained_leg_x = x + w - 76
        if retained_colors:
            swatch_w = max(3.0, 14 / len(retained_colors))
            for index, color in enumerate(retained_colors):
                parts.append(rect(retained_leg_x + index * swatch_w, y + 66, swatch_w + 0.3, bar_h, color))
        else:
            parts.append(rect(retained_leg_x, y + 66, 14, bar_h, palette['navy']))
        parts.append(text_block(x + w - 56, y + 76, ['Retained'], css_class='tiny'))
        tick_step = max(10, int(scale_max // 5) or 10)
        for tick in range(0, int(scale_max) + 1, tick_step):
            tick_x = scale_x(tick, 0, scale_max, plot_left, plot_right)
            parts.append(line(tick_x, plot_top, tick_x, plot_bottom, stroke=palette['grid'], width=1, dash='4 6'))
            parts.append(text_block(tick_x, plot_bottom + 22, [str(tick)], css_class='tiny', anchor='middle'))
        parts.append(line(plot_left, plot_bottom, plot_right, plot_bottom, stroke=palette['text'], width=1.6))
        for index, row in enumerate(rows):
            row_mid = plot_top + row_h * (index + 0.5)
            full_pct = safe_float_num(row.get('full_pct'))
            retained_pct = safe_float_num(row.get('retained_pct'))
            color = str(row.get('color'))
            full_x = scale_x(full_pct, 0, scale_max, plot_left, plot_right)
            retained_x = scale_x(retained_pct, 0, scale_max, plot_left, plot_right)
            delta_pp = retained_pct - full_pct
            # Y-axis label: use smaller font for long labels
            label_text = str(row.get('label'))
            label_cls = 'tiny' if len(label_text) > 14 else ('small' if len(label_text) > 6 else 'axis')
            parts.append(text_block(plot_left - 10, row_mid + 4, [label_text], css_class=label_cls, anchor='end'))
            # Grouped horizontal bars: Full (gray, above) + Retained (colored, below)
            full_w = max(full_x - plot_left, 1)
            retained_w = max(retained_x - plot_left, 1)
            parts.append(rect(plot_left, row_mid - bar_h - gap // 2, full_w, bar_h, full_color))
            parts.append(rect(plot_left, row_mid + gap // 2 + 1, retained_w, bar_h, color))
            # Percentage labels at bar ends
            if full_w > 30:
                parts.append(text_block(plot_left + full_w + 4, row_mid - gap // 2, [f'{full_pct:.1f}'], css_class='tiny'))
            if retained_w > 30:
                parts.append(text_block(plot_left + retained_w + 4, row_mid + gap // 2 + bar_h, [f'{retained_pct:.1f}'], css_class='tiny'))
            # Delta label
            parts.append(text_block(x + w - 14, row_mid + 4, [f'{delta_pp:+.1f}'], css_class='tiny', anchor='end', fill=color))
        return ''.join(parts)

    full_n = safe_int(summary.get('full_study_count'))
    retained_n = safe_int(summary.get('retained_study_count'))
    removed_n = safe_int(summary.get('removed_study_count'))
    full_mean = safe_float_num(tripod_summary.get('full_mean_total_score'))
    retained_mean = safe_float_num(tripod_summary.get('retained_mean_total_score'))
    full_use_case_n = safe_int(summary.get('readiness_summary', {}).get('full_use_case_n'))
    retained_use_case_n = safe_int(summary.get('readiness_summary', {}).get('retained_use_case_n'))
    retained_share = (retained_n / full_n * 100.0) if full_n else 0.0
    removed_share = (removed_n / full_n * 100.0) if full_n else 0.0
    excluded_classes = ', '.join(humanize_token(value) for value in summary.get('excluded_classes', []))
    high_tier_full = sum(safe_float_num(tier_distribution.get(tier, {}).get('full_percent')) for tier in ('II', 'III'))
    high_tier_retained = sum(safe_float_num(tier_distribution.get(tier, {}).get('retained_percent')) for tier in ('II', 'III'))
    class_counts = form_summary.get('publication_form_class_counts', {})
    retained_primary = safe_int(class_counts.get('data_rich_primary_evidence'))
    article_level_removed = safe_int(class_counts.get('reclassified_non_original_article'))
    unknown_removed = safe_int(class_counts.get('unknown'))
    class_segments = [
        ('Primary-analysis full reports', retained_primary, palette['teal']),
        ('Article-level exclusions', article_level_removed, blend(palette['red'], '#FFFFFF', 0.15)),
        ('Abstract-only', safe_int(class_counts.get('abstract_only_insufficient')), '#F3A683'),
        ('Short format', safe_int(class_counts.get('original_data_short_format')), palette['orange']),
        ('Review-like', safe_int(class_counts.get('review_like_non_original')), palette['red']),
        ('Ambiguous', unknown_removed, blend(palette['navy'], '#FFFFFF', 0.45)),
    ]
    tier_colors = {'I-a': palette['navy'], 'I-b': palette['teal'], 'II': palette['orange'], 'III': palette['red']}
    tier_rows = []
    for tier in ('I-a', 'I-b', 'II', 'III'):
        payload = tier_distribution.get(tier, {})
        tier_rows.append(
            {
                'label': tier,
                'full_pct': safe_float_num(payload.get('full_percent')),
                'retained_pct': safe_float_num(payload.get('retained_percent')),
                'color': tier_colors[tier],
            }
        )
    readiness_rows = []
    for key, label, color in [
        ('not_ready', 'Not ready', '#E64B35'),
        ('external_validation_needed', 'External validation', '#F39B7F'),
        ('human_review_only', 'Human-review-only', '#4DBBD5'),
        ('prospective_trial_candidate', 'Prospective evaluation', '#00A087'),
    ]:
        payload = readiness_dist.get(key, {})
        readiness_rows.append(
            {
                'label': label,
                'full_pct': safe_float_num(payload.get('full_percent')),
                'retained_pct': safe_float_num(payload.get('retained_percent')),
                'color': color,
            }
        )
    max_tier_pct = max((max(safe_float_num(row['full_pct']), safe_float_num(row['retained_pct'])) for row in tier_rows), default=0.0)
    max_tier_pct = max(30.0, math.ceil((max_tier_pct + 4.0) / 10.0) * 10.0)
    max_readiness_pct = max((max(safe_float_num(row['full_pct']), safe_float_num(row['retained_pct'])) for row in readiness_rows), default=0.0)
    max_readiness_pct = max(30.0, math.ceil((max_readiness_pct + 6.0) / 10.0) * 10.0)

    width, height = 1040, 600
    elements: list[str] = []

    panel_x = 15
    panel_y = 18
    panel_w = 1010
    panel_h = 150
    elements.append(rounded_box(panel_x, panel_y, panel_w, panel_h, fill='#FFFFFF', stroke=blend(palette['grid'], '#7A8797', 0.30), radius=20, extra=' class="card"'))
    elements.append(text_block(panel_x + 18, panel_y + 30, ['A'], css_class='axis', size=28, weight=700))
    elements.append(text_block(panel_x + 46, panel_y + 30, ['Report-format composition within the expanded peer-reviewed set'], css_class='axis'))
    bar_left = panel_x + 46
    bar_right = panel_x + panel_w - 46
    bar_y = panel_y + 54
    bar_h = 34
    total_bar_w = bar_right - bar_left
    cursor_x = bar_left
    for label, count, color in class_segments:
        if count <= 0:
            continue
        seg_w = total_bar_w * count / max(full_n, 1)
        elements.append(rect(cursor_x, bar_y, seg_w, bar_h, color, stroke='#FFFFFF', stroke_width=1.2))
        if seg_w >= 72:
            text_fill = '#FFFFFF' if color in {palette['navy'], palette['teal'], palette['red']} else palette['text']
            elements.append(text_block(cursor_x + seg_w / 2, bar_y + 22, [str(count)], css_class='axis', anchor='middle', fill=text_fill, size=12))
        cursor_x += seg_w
    retained_boundary = bar_left + total_bar_w * retained_n / max(full_n, 1)
    elements.append(line(retained_boundary, bar_y - 8, retained_boundary, bar_y + bar_h + 8, stroke=palette['text'], width=1.2, dash='4 4'))
    elements.append(text_block(bar_left, bar_y - 4, [f'Main set: {retained_n}/{full_n} ({retained_share:.1f}%)'], css_class='axis'))
    elements.append(text_block(bar_right - 8, bar_y - 4, [f'Removed: {removed_n} ({removed_share:.1f}%)'], css_class='axis', anchor='end'))
    legend_items = [(label, count, color) for label, count, color in class_segments if count > 0]
    for index, (label, count, color) in enumerate(legend_items):
        col = index % 3
        row = index // 3
        item_x = panel_x + 46 + col * 280
        item_y = panel_y + 102 + row * 18
        elements.append(rounded_box(item_x, item_y - 11, 14, 14, fill=color, stroke='none', radius=4))
        elements.append(text_block(item_x + 22, item_y, [f'{label} (n={count})'], css_class='small'))

    elements.append(
        dumbbell_panel(
            15,
            174,
            495,
            390,
            'B',
            'Tier distribution across denominators',
            f'II–III evidence remains dominant ({high_tier_full:.1f}%→{high_tier_retained:.1f}%).',
            tier_rows,
            max_tier_pct,
            'Share of studies (%)',
            f'study denominators {full_n} and {retained_n}',
        )
    )
    elements.append(
        dumbbell_panel(
            530,
            174,
            495,
            390,
            'C',
            'Readiness stage distribution across denominators',
            'Use-case percentages stay stable.',
            readiness_rows,
            max_readiness_pct,
            'Share of use cases (%)',
            f'use-case denominators {full_use_case_n} and {retained_use_case_n}',
        )
    )
    title = 'Figure S3. Sensitivity analysis of the main findings'
    subtitle = f'Expanded {full_n}-study peer-reviewed set versus {retained_n}-study primary peer-reviewed analysis'
    svg = svg_document(width, height, title, subtitle, ''.join(elements), palette, max_display_width=1040)
    return title, subtitle, svg


def figure_s5_yearly_maturity_detail(input_dir: pathlib.Path, palette: dict[str, str]) -> tuple[str, str, str]:
    yearly = load_yearly_maturity_summary(input_dir)
    trends = load_json(input_dir / 'statistics' / 'maturity_trends.json')
    years = [row['year'] for row in yearly]
    stage_keys = [
        ('not_ready_use_case_count', 'Not ready', '#E64B35'),
        ('external_validation_use_case_count', 'External validation', '#F39B7F'),
        ('human_review_use_case_count', 'Supervised evaluation', '#4DBBD5'),
        ('prospective_trial_use_case_count', 'Prospective evaluation', '#00A087'),
    ]
    study_counts = [safe_int(row.get('study_count')) for row in yearly]
    use_case_counts = [safe_int(row.get('use_case_count')) for row in yearly]
    high_evidence = [safe_float_num(row.get('high_evidence_share_percent')) for row in yearly]
    tripod_core = [safe_float_num(row.get('tripod_core_mean_percent')) for row in yearly]
    readiness_higher = [safe_float_num(row.get('readiness_higher_stage_share_percent')) for row in yearly]
    tier_rho = safe_float_num(trends.get('study_level', {}).get('year_to_tier_rank', {}).get('rho'))
    tripod_rho = safe_float_num(trends.get('tripod_core', {}).get('year_to_core14_score', {}).get('rho'))
    readiness_rho = safe_float_num(trends.get('readiness_use_case', {}).get('year_to_readiness_rank', {}).get('rho'))
    max_study = max(study_counts, default=1)
    max_study = int(math.ceil(max_study / 10.0) * 10)
    signal_max = max(max(high_evidence, default=0.0), max(tripod_core, default=0.0), max(readiness_higher, default=0.0))
    signal_max = max(30.0, math.ceil((signal_max + 4.0) / 10.0) * 10.0)

    def card(x: float, y: float, w: float, h: float, label: str, value: str, note: str, accent: str, fill: str) -> str:
        return ''.join(
            [
                rounded_box(x, y, w, h, fill=fill, stroke=blend(accent, '#D6DCE5', 0.35), radius=18, extra=' class="card"'),
                text_block(x + 16, y + 22, [label], css_class='small'),
                text_block(x + 16, y + 50, [value], css_class='axis', size=22),
                text_block(x + 16, y + 68, wrap_label(note, 20)[:2], css_class='tiny', line_height=13),
            ]
        )

    def panel_shell(x: float, y: float, w: float, h: float, label: str, title_text: str, subtitle_text: str) -> str:
        return ''.join(
            [
                rounded_box(x, y, w, h, fill='#FFFFFF', stroke=blend(palette['grid'], '#7A8797', 0.30), radius=20, extra=' class="card"'),
                text_block(x + 18, y + 28, [label], css_class='axis', size=16, weight=700),
                text_block(x + 46, y + 28, [title_text], css_class='axis'),
            ]
        )

    width, height = 960, 740
    elements: list[str] = []

    panel_x = 30
    panel_w = 900

    vol_y = 20
    vol_h = 190
    elements.append(panel_shell(panel_x, vol_y, panel_w, vol_h, 'A', 'Annual study volume', ''))
    plot_left = panel_x + 60
    plot_right = panel_x + panel_w - 40
    plot_top = vol_y + 78
    plot_bottom = vol_y + vol_h - 34
    tick_step = max(10, max_study // 4 or 10)
    for tick in range(0, max_study + 1, tick_step):
        y = scale_y(tick, 0, max_study, plot_bottom, plot_top)
        elements.append(line(plot_left, y, plot_right, y, stroke=palette['grid'], width=1, dash='4 6'))
        elements.append(text_block(plot_left - 10, y + 4, [str(tick)], css_class='tiny', anchor='end'))
    elements.append(line(plot_left, plot_top, plot_left, plot_bottom, stroke=palette['text'], width=1.4))
    elements.append(line(plot_left, plot_bottom, plot_right, plot_bottom, stroke=palette['text'], width=1.6))
    slot = (plot_right - plot_left) / max(len(years), 1)
    bar_w = min(80.0, slot * 0.54)
    for index, (year, count) in enumerate(zip(years, study_counts)):
        center_x = plot_left + slot * (index + 0.5)
        study_y = scale_y(count, 0, max_study, plot_bottom, plot_top)
        elements.append(rect(center_x - bar_w / 2, study_y, bar_w, plot_bottom - study_y, blend(palette['navy'], '#FFFFFF', 0.10)))
        elements.append(text_block(center_x, study_y - 8, [str(count)], css_class='small', anchor='middle'))
        elements.append(text_block(center_x, plot_bottom + 24, [str(year)], css_class='axis', anchor='middle'))

    comp_y = 230
    comp_h = 250
    elements.append(panel_shell(panel_x, comp_y, panel_w, comp_h, 'B', 'Use-case readiness composition by year', ''))
    plot_left = panel_x + 60
    plot_right = panel_x + panel_w - 40
    plot_top = comp_y + 78
    plot_bottom = comp_y + comp_h - 54
    for tick in range(0, 101, 20):
        y = scale_y(tick, 0, 100, plot_bottom, plot_top)
        elements.append(line(plot_left, y, plot_right, y, stroke=palette['grid'], width=1, dash='4 6'))
        elements.append(text_block(plot_left - 10, y + 4, [str(tick)], css_class='tiny', anchor='end'))
    elements.append(line(plot_left, plot_top, plot_left, plot_bottom, stroke=palette['text'], width=1.4))
    elements.append(line(plot_left, plot_bottom, plot_right, plot_bottom, stroke=palette['text'], width=1.6))
    slot = (plot_right - plot_left) / max(len(years), 1)
    bar_w = min(80.0, slot * 0.54)
    for index, row in enumerate(yearly):
        center_x = plot_left + slot * (index + 0.5)
        total = max(safe_int(row.get('use_case_count')), 1)
        cumulative = 0.0
        for key, _, color in stage_keys:
            count = safe_int(row.get(key))
            pct = count / total * 100.0
            y_top = scale_y(cumulative + pct, 0, 100, plot_bottom, plot_top)
            y_bottom = scale_y(cumulative, 0, 100, plot_bottom, plot_top)
            if y_bottom > y_top:
                elements.append(rect(center_x - bar_w / 2, y_top, bar_w, y_bottom - y_top, color))
                if pct >= 13.0:
                    text_fill = '#FFFFFF' if color in {'#E64B35', '#00A087'} else palette['text']
                    elements.append(text_block(center_x, y_top + (y_bottom - y_top) / 2 + 4, [f'{pct:.0f}%'], css_class='small', anchor='middle', fill=text_fill))
            cumulative += pct
        elements.append(text_block(center_x, plot_bottom + 24, [str(row.get('year'))], css_class='axis', anchor='middle'))
        elements.append(text_block(center_x, plot_bottom + 42, [f'n={total}'], css_class='tiny', anchor='middle'))
    legend_y = comp_y + 28
    legend_labels = [label for _, label, _ in stage_keys]
    legend_gap = 6
    label_unit = 6.1
    swatch_gap = 20
    total_legend_w = sum(len(lb) * label_unit + swatch_gap for lb in legend_labels) + legend_gap * (len(legend_labels) - 1)
    legend_x0 = panel_x + panel_w - 18 - total_legend_w
    cursor_x = legend_x0
    for _, label, color in stage_keys:
        elements.append(rect(cursor_x, legend_y - 11, 15, 15, color))
        elements.append(text_block(cursor_x + 22, legend_y, [label], css_class='small'))
        cursor_x += len(label) * label_unit + swatch_gap + legend_gap

    trend_y = 500
    trend_h = 230
    elements.append(panel_shell(panel_x, trend_y, panel_w, trend_h, 'C', 'Descriptive temporal trends', 'Evidence quality improves over time, but most use cases remain below deployment-ready stages.'))
    plot_left = panel_x + 60
    plot_right = panel_x + panel_w - 40
    plot_top = trend_y + 78
    plot_bottom = trend_y + trend_h - 68
    for tick in range(0, int(signal_max) + 1, 10):
        y = scale_y(tick, 0, signal_max, plot_bottom, plot_top)
        elements.append(line(plot_left, y, plot_right, y, stroke=palette['grid'], width=1, dash='4 6'))
        elements.append(text_block(plot_left - 10, y + 4, [str(tick)], css_class='tiny', anchor='end'))
    elements.append(line(plot_left, plot_top, plot_left, plot_bottom, stroke=palette['text'], width=1.4))
    elements.append(line(plot_left, plot_bottom, plot_right, plot_bottom, stroke=palette['text'], width=1.6))
    x_positions = {year: plot_left + ((plot_right - plot_left) / max(len(years), 1)) * (index + 0.5) for index, year in enumerate(years)}
    signal_specs = [
        ('High-evidence', high_evidence, palette['navy'], years),
        ('Higher-stage', readiness_higher, palette['orange'], years),
        ('Core-14', tripod_core, palette['teal'], years),
    ]
    signal_label_positions: list[tuple[float, float, str, str]] = []
    for label, values, color, years_subset in signal_specs:
        points = [(x_positions[year], scale_y(value, 0, signal_max, plot_bottom, plot_top)) for year, value in zip(years_subset, values)]
        elements.append(polyline(points, stroke=color, width=3.4))
        for year, value in zip(years_subset, values):
            x = x_positions[year]
            y = scale_y(value, 0, signal_max, plot_bottom, plot_top)
            elements.append(circle(x, y, 5.0, color, stroke='#FFFFFF', stroke_width=1.6))
        last_x, last_y = points[-1]
        signal_label_positions.append((last_x, last_y, f'{label} {values[-1]:.1f}%', color))
    # De-overlap labels: sort by y, enforce minimum 14px vertical gap
    signal_label_positions.sort(key=lambda t: t[1])
    adjusted_ys: list[float] = []
    for i, (lx, ly, txt, clr) in enumerate(signal_label_positions):
        desired_y = ly - 10
        if adjusted_ys and desired_y < adjusted_ys[-1] + 14:
            desired_y = adjusted_ys[-1] + 14
        adjusted_ys.append(desired_y)
    for (lx, ly, txt, clr), adj_y in zip(signal_label_positions, adjusted_ys):
        elements.append(text_block(lx + 12, adj_y, [txt], css_class='small'))
    for year in years:
        elements.append(text_block(x_positions[year], plot_bottom + 24, [str(year)], css_class='axis', anchor='middle'))

    title = 'Figure S4. Year-specific maturity profile'
    if years:
        subtitle = f'Study volume, readiness composition, and temporal trends from {years[0]} through early {years[-1]}'
    else:
        subtitle = 'Study volume, readiness composition, and temporal trends by year'
    svg = svg_document(width, height, title, subtitle, ''.join(elements), palette)
    return title, subtitle, svg


def figure_s6_protocol_ab_comparison(input_dir: pathlib.Path, palette: dict[str, str]) -> tuple[str, str, str]:
    rows = load_csv(input_dir / 'protocol_comparison' / 'comparison_table.csv')
    submission_summary_path = input_dir / 'protocol_comparison' / 'submission_facing_summary.json'
    if submission_summary_path.exists():
        comparison_summary = load_json(submission_summary_path)
        protocol_a_total = safe_int(comparison_summary.get('expanded_peer_reviewed_comparison_set_count'))
        protocol_b_total = safe_int(comparison_summary.get('supplementary_publication_status_records_reviewed'))
        protocol_b_unique = safe_int(comparison_summary.get('supplementary_publication_status_records_unique'))
        protocol_b_overlap = safe_int(comparison_summary.get('supplementary_publication_status_records_overlap'))
    else:
        comparison_summary = load_json(input_dir / 'protocol_comparison' / 'comparison_summary.json')
        protocol_a_total = safe_int(comparison_summary.get('protocol_a_total'))
        protocol_b_total = safe_int(comparison_summary.get('protocol_b_total'))
        protocol_b_unique = safe_int(comparison_summary.get('protocol_b_unique_records'))
        protocol_b_overlap = safe_int((comparison_summary.get('overlap') or {}).get('overlap'))
    tier_rows = [row for row in rows if row.get('domain') == 'tier']
    crl_rows = [row for row in rows if row.get('domain') == 'crl']
    tripod_row = next((row for row in rows if row.get('domain') == 'tripod_mean'), None)

    width, height = 1080, 660
    elements: list[str] = []
    strip_top = 14
    strip_bottom = 84
    strip_left = 30
    strip_right = 1050
    tier_y = 40
    elements.append(line(30, tier_y, 1050, tier_y, stroke=palette['navy'], width=2))
    elements.append(text_block(30, tier_y + 22, ['Tier counts: expanded peer-reviewed comparison set vs supplementary publication-status stream'], css_class='axis'))

    left_x0, left_x1 = 80, 1015
    left_y0, left_y1 = 136, 348
    max_tier = max((max(safe_int(row.get('protocol_a')), safe_int(row.get('protocol_b'))) for row in tier_rows), default=1)
    max_tier = int(math.ceil(max_tier / 10.0) * 10)
    elements.append(line(left_x0, left_y1, left_x1, left_y1, stroke=palette['text'], width=2))
    elements.append(line(left_x0, left_y0, left_x0, left_y1, stroke=palette['text'], width=2))
    for tick in range(0, max_tier + 1, max(10, max_tier // 5 or 1)):
        y = scale_y(tick, 0, max_tier, left_y1, left_y0)
        elements.append(line(left_x0, y, left_x1, y, stroke=palette['grid'], width=1, dash='4 6'))
        elements.append(text_block(left_x0 - 12, y + 4, [str(tick)], css_class='small', anchor='end'))

    slot = (left_x1 - left_x0) / max(len(tier_rows), 1)
    bar_w = max(26.0, slot * 0.18)
    for index, row in enumerate(tier_rows):
        center_x = left_x0 + slot * (index + 0.5)
        value_a = safe_int(row.get('protocol_a'))
        value_b = safe_int(row.get('protocol_b'))
        y_a = scale_y(value_a, 0, max_tier, left_y1, left_y0)
        y_b = scale_y(value_b, 0, max_tier, left_y1, left_y0)
        elements.append(rect(center_x - bar_w - 4, y_a, bar_w, left_y1 - y_a, palette['navy']))
        elements.append(rect(center_x + 4, y_b, bar_w, left_y1 - y_b, palette['teal']))
        elements.append(text_block(center_x, left_y1 + 34, [row.get('category', '')], css_class='axis', anchor='middle'))
        elements.append(text_block(center_x - bar_w / 2 - 4, y_a - 8, [str(value_a)], css_class='tiny', anchor='middle'))
        elements.append(text_block(center_x + bar_w / 2 + 4, y_b - 8, [str(value_b)], css_class='tiny', anchor='middle'))

    legend_y = 392
    elements.append(rect(100, legend_y, 14, 14, palette['navy']))
    elements.append(text_block(120, legend_y + 12, [f'Expanded peer-reviewed set (n={protocol_a_total})'], css_class='small'))
    elements.append(rect(540, legend_y, 14, 14, palette['teal']))
    elements.append(text_block(560, legend_y + 12, [f'Supplementary publication-status stream (n={protocol_b_total})'], css_class='small'))

    lower_y = 426
    elements.append(line(30, lower_y, 1050, lower_y, stroke=palette['orange'], width=2))
    elements.append(text_block(30, lower_y + 22, ['Clinical risk level counts and TRIPOD summary'], css_class='axis'))

    right_x0, right_x1 = 200, 500
    right_y0, right_y1 = lower_y + 70, lower_y + 170
    max_crl = max((max(safe_int(row.get('protocol_a')), safe_int(row.get('protocol_b'))) for row in crl_rows), default=1)
    max_crl = int(math.ceil(max_crl / 10.0) * 10)
    elements.append(line(right_x0, right_y1, right_x1, right_y1, stroke=palette['text'], width=2))
    for tick in range(0, max_crl + 1, max(10, max_crl // 4 or 1)):
        x = scale_x(tick, 0, max_crl, right_x0, right_x1)
        elements.append(line(x, right_y0, x, right_y1, stroke=palette['grid'], width=1, dash='4 6'))
        elements.append(text_block(x, right_y1 + 24, [str(tick)], css_class='tiny', anchor='middle'))
    slot_h = (right_y1 - right_y0) / max(len(crl_rows), 1)
    for index, row in enumerate(crl_rows):
        row_mid = right_y0 + slot_h * (index + 0.5)
        value_a = safe_int(row.get('protocol_a'))
        value_b = safe_int(row.get('protocol_b'))
        x_a = scale_x(value_a, 0, max_crl, right_x0, right_x1)
        x_b = scale_x(value_b, 0, max_crl, right_x0, right_x1)
        elements.append(text_block(30, row_mid + 4, [row.get('category', '')], css_class='axis'))
        elements.append(line(min(x_a, x_b), row_mid, max(x_a, x_b), row_mid, stroke=palette['grid'], width=3))
        elements.append(circle(x_a, row_mid, 5, '#ffffff', stroke=palette['navy'], stroke_width=2))
        elements.append(circle(x_b, row_mid, 5, palette['teal'], stroke='#ffffff', stroke_width=1.5))
        elements.append(text_block(x_a + 8, row_mid - 5, [f'A {value_a}'], css_class='tiny'))
        elements.append(text_block(x_b + 8, row_mid + 12, [f'B {value_b}'], css_class='tiny'))

    tripod_a = safe_float_num(tripod_row.get('protocol_a')) if tripod_row else 0.0
    tripod_b = safe_float_num(tripod_row.get('protocol_b')) if tripod_row else 0.0
    delta = tripod_b - tripod_a
    stats_x = 540
    elements.append(text_block(stats_x, lower_y + 50, ['TRIPOD compliance-rate mean'], css_class='axis'))
    elements.append(rounded_box(stats_x, lower_y + 66, 170, 90, fill='#F5FAFF', stroke=palette['navy']))
    elements.append(rounded_box(stats_x + 190, lower_y + 66, 260, 90, fill='#F7FCF8', stroke=palette['teal']))
    elements.append(text_block(stats_x + 14, lower_y + 88, ['Expanded peer-reviewed set'], css_class='small'))
    elements.append(text_block(stats_x + 14, lower_y + 124, [f'{tripod_a:.2f}%'], css_class='title', size=22))
    elements.append(text_block(stats_x + 204, lower_y + 88, ['Supplementary publication-status stream'], css_class='small'))
    elements.append(text_block(stats_x + 204, lower_y + 124, [f'{tripod_b:.2f}%'], css_class='title', size=22))
    elements.append(
        text_block(
            stats_x,
            lower_y + 174,
            [f'Δ {delta:+.2f} pp between the two study streams'],
            css_class='small',
            line_height=16,
        )
    )

    title = 'Figure S5. Peer-reviewed versus supplementary publication-status source comparison'
    subtitle = (
        f'Descriptive tier, clinical risk level, and reporting comparisons between the expanded {protocol_a_total}-study '
        'peer-reviewed comparison set and the supplementary publication-status source stream after concept-level overlap review'
    )
    svg = svg_document(width, height, title, subtitle, ''.join(elements), palette, max_display_width=1060)
    return title, subtitle, svg


def figure_s7_sample_size_distribution(input_dir: pathlib.Path, palette: dict[str, str]) -> tuple[str, str, str]:
    rows = load_csv(input_dir / 'statistics' / 'sample_size_summary.csv')
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        parsed_rows.append(
            {
                'tier': row.get('tier', ''),
                'gi_subsite': row.get('gi_subsite', ''),
                'n': safe_int(row.get('n')),
                'median': safe_float_num(row.get('median')),
                'lt_100_ratio': safe_float_num(row.get('lt_100_ratio')),
            }
        )
    positive_medians = [max(entry['median'], 1.0) for entry in parsed_rows]
    x_max = max(positive_medians, default=10.0)
    x_ticks = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    x_ticks = [tick for tick in x_ticks if tick <= x_max * 1.1]
    if not x_ticks or x_ticks[-1] < x_max:
        x_ticks.append(int(math.ceil(x_max / 100.0) * 100.0))

    tier_agg: dict[str, dict[str, float]] = {}
    for tier in TIERS:
        tier_rows = [entry for entry in parsed_rows if entry['tier'] == tier]
        total_n = sum(entry['n'] for entry in tier_rows)
        if total_n > 0:
            weighted_ratio = sum(entry['lt_100_ratio'] * entry['n'] for entry in tier_rows) / total_n
            weighted_median = sum(entry['median'] * entry['n'] for entry in tier_rows) / total_n
        else:
            weighted_ratio = 0.0
            weighted_median = 0.0
        tier_agg[tier] = {
            'total_n': float(total_n),
            'weighted_ratio': weighted_ratio * 100.0,
            'weighted_median': weighted_median,
        }

    width = 1060
    elements = [
        text_block(70, 34, ['Tier/subsite strata'], css_class='axis'),
        text_block(470, 34, ['Tier-level weighted summary'], css_class='axis'),
    ]

    # ── Left bubble panel ────────────────────────────────────────────
    left_x0, left_x1 = 70, 420
    left_y0, left_y1 = 56, 300
    elements.append(line(left_x0, left_y1, left_x1, left_y1, stroke=palette['text'], width=1.4))
    elements.append(line(left_x0, left_y0, left_x0, left_y1, stroke=palette['text'], width=1.4))
    log_min = math.log10(1.0)
    log_max = math.log10(max(x_ticks))
    for tick in x_ticks:
        x = scale_x(math.log10(tick), log_min, log_max, left_x0, left_x1)
        elements.append(line(x, left_y0, x, left_y1, stroke=palette['grid'], width=1, dash='4 6'))
        elements.append(text_block(x, left_y1 + 20, [str(tick)], css_class='tiny', anchor='middle'))
    for tick in range(0, 101, 20):
        y = scale_y(tick, 0, 100, left_y1, left_y0)
        elements.append(line(left_x0, y, left_x1, y, stroke=palette['grid'], width=1, dash='4 6'))
        elements.append(text_block(left_x0 - 8, y + 4, [str(tick)], css_class='tiny', anchor='end'))

    for entry in parsed_rows:
        x = scale_x(math.log10(max(entry['median'], 1.0)), log_min, log_max, left_x0, left_x1)
        y = scale_y(entry['lt_100_ratio'] * 100.0, 0, 100, left_y1, left_y0)
        radius = max(4.0, min(16.0, math.sqrt(max(entry['n'], 1)) * 1.5))
        subsite_color = SUBSITE_COLORS.get(entry['gi_subsite'], palette['gray'])
        elements.append(circle(x, y, radius, subsite_color, stroke='#ffffff', stroke_width=1.2))

    elements.append(text_block(left_x0, left_y1 + 42, ['Median sample size (log scale)'], css_class='small'))
    # Y-axis label — vertical, centered along left axis
    y_label_mid = (left_y0 + left_y1) / 2
    elements.append(f'<text x="{left_x0 - 36}" y="{y_label_mid}" class="small" text-anchor="middle" '
                    f'transform="rotate(-90,{left_x0 - 36},{y_label_mid})">'
                    '&lt;100 ratio (%)</text>')

    # ── Right tier summary panel ─────────────────────────────────────
    bar_x0, bar_x1 = 560, 1020
    bar_y0, bar_y1 = 60, 300
    max_summary_n = max((tier_agg[tier]['total_n'] for tier in TIERS), default=1.0)
    max_summary_n = max(1.0, math.ceil(max_summary_n / 10.0) * 10.0)
    slot_h = (bar_y1 - bar_y0) / max(len(TIERS), 1)
    for tick in range(0, int(max_summary_n) + 1, max(10, int(max_summary_n // 4) or 1)):
        x = scale_x(tick, 0, max_summary_n, bar_x0, bar_x1)
        elements.append(line(x, bar_y0, x, bar_y1, stroke=palette['grid'], width=1, dash='4 6'))
        elements.append(text_block(x, bar_y1 + 18, [str(tick)], css_class='tiny', anchor='middle'))
    elements.append(line(bar_x0, bar_y1, bar_x1, bar_y1, stroke=palette['text'], width=1.4))
    for index, tier in enumerate(TIERS):
        row_mid = bar_y0 + slot_h * (index + 0.5)
        payload = tier_agg[tier]
        x_end = scale_x(payload['total_n'], 0, max_summary_n, bar_x0, bar_x1)
        tier_color = {
            'S': palette['purple'],
            'I-a': palette['navy'],
            'I-b': palette['teal'],
            'II': palette['orange'],
            'III': palette['red'],
        }.get(tier, palette['gray'])
        elements.append(text_block(470, row_mid + 3, [tier], css_class='axis'))
        elements.append(line(bar_x0, row_mid - 4, x_end, row_mid - 4, stroke=tier_color, width=6))
        elements.append(text_block(x_end + 6, row_mid, [f"n={int(payload['total_n'])}"], css_class='tiny'))
        elements.append(text_block(470, row_mid + 16, [f"<100: {payload['weighted_ratio']:.1f}%"], css_class='small'))
        elements.append(text_block(470, row_mid + 30, [f"Med: {payload['weighted_median']:.1f}"], css_class='small'))

    # ── Legends below both panels ────────────────────────────────────
    leg_y_start = max(left_y1, bar_y1) + 57

    # Bubble size legend — left-aligned, on top
    bubble_y = leg_y_start
    for idx, sample_n in enumerate([4, 8, 11]):
        cx = left_x0 + 16 + idx * 120
        radius = max(4.0, min(16.0, math.sqrt(sample_n) * 1.5))
        elements.append(circle(cx, bubble_y, radius, '#D9E6F5', stroke=palette['navy'], stroke_width=1.0))
        elements.append(text_block(cx + 22, bubble_y + 4, [f'n\u2248{sample_n}'], css_class='small'))

    # Subsite legend — centered across full width
    legend_items = [subsite for subsite in GI_SUBSITE_ORDER if any(entry['gi_subsite'] == subsite for entry in parsed_rows)]
    item_w = 140
    cols_per_row = min(len(legend_items), 5) or 5
    legend_row_count = ((len(legend_items) - 1) // cols_per_row + 1) if legend_items else 1
    total_legend_w = cols_per_row * item_w
    leg_x0 = (width - total_legend_w) / 2
    subsite_y_start = bubble_y + 20
    for index, subsite in enumerate(legend_items):
        row_i = index // cols_per_row
        col_i = index % cols_per_row
        lx = leg_x0 + col_i * item_w
        ly = subsite_y_start + row_i * 22
        elements.append(rect(lx, ly, 10, 10, SUBSITE_COLORS.get(subsite, palette['gray'])))
        elements.append(text_block(lx + 16, ly + 10, [subsite.replace('_', ' ')], css_class='small'))

    height = int(subsite_y_start + legend_row_count * 22 + 20)
    title = 'Figure S6. Sample-size distribution'
    subtitle = 'Stratum-level sample-size distribution and tier-level small-sample diagnostics'
    svg = svg_document(width, height, title, subtitle, ''.join(elements), palette)
    return title, subtitle, svg


def figure_s8_readiness_validation_audit(input_dir: pathlib.Path, palette: dict[str, str]) -> tuple[str, str, str]:
    validated_use_case_path = input_dir / 'readiness' / 'readiness_validation_full_97_2026-04-06.csv'
    agreement_summary_path = input_dir / 'readiness' / 'readiness_validation_agreement_summary_2026-04-06.csv'
    use_case_rows: list[dict[str, Any]]
    agreement_row: dict[str, Any] = {}
    review_status_value = 'completed'
    review_status_note = 'dual review + adjudication'

    if validated_use_case_path.exists():
        use_case_rows = load_csv(validated_use_case_path)
        if agreement_summary_path.exists():
            agreement_rows = load_csv(agreement_summary_path)
            if agreement_rows:
                agreement_row = agreement_rows[0]
    else:
        # Fallback for legacy rebuilds only.
        use_case_rows = load_csv(input_dir / 'readiness' / 'validation_sample_use_case_level.csv')
        review_status_value = 'limited'
        review_status_note = 'full adjudication file unavailable'

    per_study_tier: dict[str, str] = {}
    for row in use_case_rows:
        record_id = str(row.get('record_id') or '').strip()
        if not record_id:
            continue
        if record_id not in per_study_tier:
            per_study_tier[record_id] = str(row.get('tier') or 'unknown').strip()
    study_tier_counts = Counter(per_study_tier.values())
    use_stage_counts = Counter(
        (row.get('adjudicated_final_stage') or row.get('readiness_stage') or 'unknown').strip()
        for row in use_case_rows
    )

    study_total = len(per_study_tier)
    use_total = len(use_case_rows)
    reviewed_use_cases = safe_int(agreement_row.get('adjudicated_completed_n', use_total))
    percent_agreement = safe_float_num(agreement_row.get('pre_adjudication_percent_agreement'))
    kappa = safe_float_num(agreement_row.get('cohens_kappa'))
    discordant_n = safe_int(agreement_row.get('pre_adjudication_discordant_n'))

    width = 1200  # widened to preserve right-side value labels in submission renders
    elements: list[str] = []
    sl, sr = 40, 1160

    # ── Study-level tier bar section ─────────────────────────────────
    chart_left, chart_right = 380, 1080
    row_h = 36
    section_y = 20
    elements.append(line(sl, section_y, sr, section_y, stroke=palette['navy'], width=1.5))
    elements.append(text_block(sl, section_y + 20, ['Study-level distribution in adjudicated validation set'], css_class='axis'))

    tier_bar_specs = [
        ('S', 'S', palette['purple']),
        ('I-a', 'I-a', palette['navy']),
        ('I-b', 'I-b', palette['teal']),
        ('II', 'II', palette['orange']),
        ('III', 'III', palette['red']),
    ]
    chart_top = section_y + 34
    chart_bottom = chart_top + len(tier_bar_specs) * row_h
    total_study = sum(study_tier_counts.get(key, 0) for key, _, _ in tier_bar_specs)
    represented_tiers = sum(1 for key, _, _ in tier_bar_specs if study_tier_counts.get(key, 0) > 0)
    max_study = max((study_tier_counts.get(key, 0) for key, _, _ in tier_bar_specs), default=1)
    max_study = int(math.ceil(max_study / 5.0) * 5) or 5
    tick_step = max(5, max_study // 4 or 5)
    for tick in range(0, max_study + 1, tick_step):
        x = scale_x(tick, 0, max_study, chart_left, chart_right)
        elements.append(line(x, chart_top, x, chart_bottom, stroke=palette['grid'], width=1, dash='4 6'))
        elements.append(text_block(x, chart_bottom + 14, [str(tick)], css_class='tiny', anchor='middle'))
    elements.append(line(chart_left, chart_top, chart_left, chart_bottom, stroke=palette['text'], width=1.4))
    elements.append(line(chart_left, chart_bottom, chart_right, chart_bottom, stroke=palette['text'], width=1.4))
    for i, (key, label, color) in enumerate(tier_bar_specs):
        y = chart_top + i * row_h
        value = study_tier_counts.get(key, 0)
        pct = (value / total_study * 100.0) if total_study else 0.0
        bar_h = row_h - 8
        bar_y = y + 4
        bar_w = scale_x(value, 0, max_study, chart_left, chart_right) - chart_left
        elements.append(rect(chart_left, bar_y, max(bar_w, 0), bar_h, color))
        elements.append(text_block(chart_left - 8, bar_y + bar_h / 2 + 4, [label], css_class='small', anchor='end'))
        elements.append(text_block(chart_left + max(bar_w, 0) + 6, bar_y + bar_h / 2 + 4, [f'{value} ({pct:.0f}%)'], css_class='tiny'))

    # ── Use-case readiness-stage bar section ─────────────────────────
    section2_y = chart_bottom + 34
    elements.append(line(sl, section2_y, sr, section2_y, stroke='#8491B4', width=1.5))
    elements.append(text_block(sl, section2_y + 20, ['Use-case distribution after adjudication'], css_class='axis'))

    stage_bar_specs = [
        ('not_ready', 'Not ready for clinical use', '#E64B35'),
        ('external_validation_needed', 'External-validation candidates', '#F39B7F'),
        ('human_review_only', 'Low-risk, human-review-only', '#4DBBD5'),
        ('prospective_trial_candidate', 'Prospective-trial candidates', '#00A087'),
    ]
    chart_top2 = section2_y + 34
    chart_bottom2 = chart_top2 + len(stage_bar_specs) * row_h
    total_use = sum(use_stage_counts.get(key, 0) for key, _, _ in stage_bar_specs)
    represented_stages = sum(1 for key, _, _ in stage_bar_specs if use_stage_counts.get(key, 0) > 0)
    max_use = max((use_stage_counts.get(key, 0) for key, _, _ in stage_bar_specs), default=1)
    max_use = int(math.ceil(max_use / 5.0) * 5) or 5
    tick_step2 = max(5, max_use // 4 or 5)
    for tick in range(0, max_use + 1, tick_step2):
        x = scale_x(tick, 0, max_use, chart_left, chart_right)
        elements.append(line(x, chart_top2, x, chart_bottom2, stroke=palette['grid'], width=1, dash='4 6'))
        elements.append(text_block(x, chart_bottom2 + 14, [str(tick)], css_class='tiny', anchor='middle'))
    elements.append(line(chart_left, chart_top2, chart_left, chart_bottom2, stroke=palette['text'], width=1.4))
    elements.append(line(chart_left, chart_bottom2, chart_right, chart_bottom2, stroke=palette['text'], width=1.4))
    for i, (key, label, color) in enumerate(stage_bar_specs):
        y = chart_top2 + i * row_h
        value = use_stage_counts.get(key, 0)
        pct = (value / total_use * 100.0) if total_use else 0.0
        bar_h = row_h - 8
        bar_y = y + 4
        bar_w = scale_x(value, 0, max_use, chart_left, chart_right) - chart_left
        elements.append(rect(chart_left, bar_y, max(bar_w, 0), bar_h, color))
        elements.append(text_block(chart_left - 8, bar_y + bar_h / 2 + 4, [label], css_class='small', anchor='end'))
        elements.append(text_block(chart_left + max(bar_w, 0) + 6, bar_y + bar_h / 2 + 4, [f'{value} ({pct:.0f}%)'], css_class='tiny'))

    height = int(chart_bottom2 + 30)
    title = 'Figure S7. Validation agreement and adjudication summary'
    if percent_agreement and kappa:
        subtitle = f'Full dual-review adjudication across {use_total} use cases (agreement {percent_agreement:.1f}%, kappa {kappa:.3f})'
    else:
        subtitle = f'Use-case-level adjudication summary across {use_total} validated use cases'
    svg = svg_document(width, height, title, subtitle, ''.join(elements), palette, max_display_width=1160)
    return title, subtitle, svg


def figure_s8_study_level_forest(input_dir: pathlib.Path, palette: dict[str, str]) -> tuple[str, str, str]:
    meta_rows = load_csv(input_dir / 'meta' / 'meta_study_table.csv')
    family_results = load_json(input_dir / 'meta' / 'family_results.json')
    rows_by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in meta_rows:
        rows_by_family[row.get('metric_family', 'other')].append(row)
    family_order = [
        family for family in ['accuracy', 'sensitivity', 'auc_auroc', 'f1', 'specificity']
        if len(rows_by_family.get(family, [])) >= 2
    ]
    if not family_order:
        family_order = [family for family in ['accuracy', 'sensitivity', 'auc_auroc', 'f1', 'specificity'] if rows_by_family.get(family)]

    candidates = []
    for row in meta_rows:
        effect = safe_float_num(row.get('effect_size'))
        se = safe_float_num(row.get('se'))
        candidates.extend([effect, max(0.0, effect - 1.96 * se), min(1.0, effect + 1.96 * se)])
    x_min = min(0.3, math.floor(min(candidates or [0.3]) * 10) / 10)
    x_max = max(1.0, math.ceil(max(candidates or [1.0]) * 10) / 10)

    plot_left, plot_right = 440, 900
    estimate_x = 1060
    top = 20
    row_height = 28
    row_y = top + 76
    alt_row_fill = '#F7F9FC'
    study_color = palette['navy']
    family_colors = {'accuracy': palette['navy'], 'sensitivity': palette['teal'], 'auc_auroc': palette['orange'], 'f1': '#4DBBD5', 'specificity': palette['purple']}
    results_by_family = family_results.get('results', {})

    # ── First pass: collect row positions for alternating backgrounds ──
    row_entries: list[dict[str, Any]] = []
    scan_y = row_y
    for family in family_order:
        family_rows = sorted(rows_by_family.get(family, []), key=lambda item: safe_float_num(item.get('effect_size')), reverse=True)
        scan_y += row_height  # family header row
        for _row in family_rows:
            row_entries.append({'y': scan_y, 'type': 'study'})
            scan_y += row_height
        scan_y += 10

    elements: list[str] = []

    # ── Alternating row backgrounds (drawn first, behind everything) ──
    global_idx = 0
    for entry in row_entries:
        if global_idx % 2 == 1:
            elements.append(rect(30, entry['y'] - row_height / 2 - 2, 1040, row_height, alt_row_fill))
        global_idx += 1

    # ── Column headers ───────────────────────────────────────────────
    elements.append(text_block(40, top + 18, ['Study or family'], css_class='axis'))
    elements.append(text_block((plot_left + plot_right) / 2, top + 18, ['Effect size and 95% CI'], css_class='axis', anchor='middle'))
    elements.append(text_block(estimate_x, top + 18, ['Estimate [95% CI]'], css_class='axis', anchor='end'))
    elements.append(line(30, top + 28, 1070, top + 28, stroke=palette['text'], width=1.5))

    # ── Tick axis ────────────────────────────────────────────────────
    axis_y = top + 34
    for tick_index in range(int(x_min * 10), int(x_max * 10) + 1):
        tick = tick_index / 10
        x = scale_x(tick, x_min, x_max, plot_left, plot_right)
        elements.append(line(x, axis_y, x, axis_y + 8, stroke=palette['text'], width=1.5))
        elements.append(text_block(x, axis_y + 28, [f'{tick:.1f}'], css_class='small', anchor='middle'))

    for family in family_order:
        family_name = family_label(family)
        fam_color = family_colors.get(family, palette['navy'])
        family_rows = sorted(rows_by_family.get(family, []), key=lambda item: safe_float_num(item.get('effect_size')), reverse=True)
        family_payload = results_by_family.get(family) or {}
        heterogeneity_text = heterogeneity_note(
            family_payload.get('heterogeneity_result') or family_payload.get('pooled_result')
        )

        # Family header — bold, with a subtle colored left accent
        elements.append(line(34, row_y - 8, 34, row_y + 6, stroke=fam_color, width=3))
        elements.append(text_block(44, row_y + 4, [f'{family_name} (n = {len(family_rows)})'], css_class='axis'))
        if heterogeneity_text:
            elements.append(text_block(estimate_x, row_y + 4, [heterogeneity_text], css_class='small', anchor='end'))
        row_y += row_height

        for row in family_rows:
            effect = safe_float_num(row.get('effect_size'))
            se = safe_float_num(row.get('se'))
            low = max(0.0, effect - 1.96 * se)
            high = min(1.0, effect + 1.96 * se)
            x_low = scale_x(low, x_min, x_max, plot_left, plot_right)
            x_high = scale_x(high, x_min, x_max, plot_left, plot_right)
            x_mid = scale_x(effect, x_min, x_max, plot_left, plot_right)
            box_size = max(5, min(11, math.sqrt(max(safe_int(row.get('sample_size'), 100), 25)) / 2.2))

            elements.append(text_block(44, row_y + 4, [forest_study_label(row)], css_class='body'))
            # CI line and whisker caps — uniform navy
            elements.append(line(x_low, row_y, x_high, row_y, stroke=study_color, width=1.8))
            elements.append(line(x_low, row_y - 6, x_low, row_y + 6, stroke=study_color, width=1.5))
            elements.append(line(x_high, row_y - 6, x_high, row_y + 6, stroke=study_color, width=1.5))
            # Study square — uniform navy
            elements.append(rect(x_mid - box_size / 2, row_y - box_size / 2, box_size, box_size,
                                 study_color, stroke='#ffffff', stroke_width=1.0))
            elements.append(text_block(estimate_x, row_y + 4, [fmt_ci(effect, low, high)], css_class='small', anchor='end'))
            row_y += row_height
        row_y += 10

    # ── Bottom rule ──────────────────────────────────────────────────
    elements.append(line(30, row_y - 12, 1070, row_y - 12, stroke=palette['text'], width=1.5))

    height = max(600, int(row_y + 10))
    title = 'Figure S8. Expanded study-level forest display for accuracy-like endpoints'
    subtitle = 'Accuracy-like study-level forest; single-study sensitivity and F1 endpoints are summarized textually'
    svg = svg_document(1100, height, title, subtitle, ''.join(elements), palette)
    return title, subtitle, svg


def write_manifest(
    paths: list[pathlib.Path],
    output_path: pathlib.Path,
    *,
    input_dir: pathlib.Path,
    figures_dir: pathlib.Path,
) -> None:
    files: list[str] = []
    for path in paths:
        try:
            files.append(str(path.relative_to(figures_dir)))
        except ValueError:
            files.append(str(path))
    main_count = len({path.stem for path in paths if path.parent.name == 'main'})
    supplementary_count = len({path.stem for path in paths if path.parent.name == 'supplementary'})
    report = {
        'input_dir': str(input_dir),
        'output_dir': str(figures_dir),
        'main_count': main_count,
        'supplementary_count': supplementary_count,
        'files': files,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def main() -> None:
    args = parse_args()
    input_dir = pathlib.Path(args.input_dir)
    figures_dir = pathlib.Path(args.figures_dir) if args.figures_dir else input_dir / 'figures'
    supplementary_dir = pathlib.Path(args.supplementary_dir) if args.supplementary_dir else input_dir / 'supplementary'
    palette = dict(STYLE_PRESETS[args.style])
    palette['text'] = '#000000'

    tier_csv = find_tier_csv(input_dir)
    if tier_csv is None:
        raise SystemExit('No tier-labeled CSV found under input-dir/tier_labeled/')
    records = normalize_records(load_csv(tier_csv))
    if not records:
        raise SystemExit('No records available for HTML/SVG figure generation')

    generated_paths: list[pathlib.Path] = []

    # ── Main figures ──────────────────────────────────────────────────
    main_dir = figures_dir / 'main'
    main_dir.mkdir(parents=True, exist_ok=True)
    main_figures = [
        ('figure_1_prisma_flow', figure_one_prisma(input_dir, palette)),
        ('figure_2_tier_crl_workflow', figure_two_three_combined(records, palette)),
        ('figure_3_volume_maturity_trends', figure_five_temporal(input_dir, palette)),
        ('figure_4_tripod_reporting_profile', figure_six_tripod(input_dir, palette)),
        ('figure_5_clinical_readiness_roadmap', figure_seven_readiness(input_dir, palette)),
        ('figure_6_forest_plot', figure_eight_forest(input_dir, palette)),
    ]
    for stem, (title, subtitle, svg_markup) in main_figures:
        path_without_suffix = main_dir / stem
        save_dual_output(path_without_suffix, title, subtitle, svg_markup)
        generated_paths.extend([path_without_suffix.with_suffix('.svg'), path_without_suffix.with_suffix('.html')])

    # ── Supplementary figures ─────────────────────────────────────────
    supplementary_figures = [
        ('figure_s1_model_ecosystem', figure_s1_bayesian_and_ecosystem(input_dir, records, palette)),
        ('figure_s2_doi_plot', figure_s3_doi(input_dir, palette)),
        ('figure_s3_publication_form_sensitivity', figure_s4_publication_form_sensitivity(input_dir, palette)),
        ('figure_s4_yearly_maturity_detail', figure_s5_yearly_maturity_detail(input_dir, palette)),
        ('figure_s5_stream_comparison', figure_s6_protocol_ab_comparison(input_dir, palette)),
        ('figure_s6_sample_size_distribution', figure_s7_sample_size_distribution(input_dir, palette)),
        ('figure_s7_readiness_validation_audit', figure_s8_readiness_validation_audit(input_dir, palette)),
        ('figure_s8_study_level_forest', figure_s8_study_level_forest(input_dir, palette)),
    ]
    legacy_supplementary_stems = [
        'figure_s1_bayesian_and_ecosystem',
        'figure_s5_protocol_ab_comparison',
        'figure_s8_full_forest_legacy_support',
    ]
    for stem, (title, subtitle, svg_markup) in supplementary_figures:
        path_without_suffix = supplementary_dir / stem
        save_dual_output(path_without_suffix, title, subtitle, svg_markup)
        generated_paths.extend([path_without_suffix.with_suffix('.svg'), path_without_suffix.with_suffix('.html')])
    for legacy_stem in legacy_supplementary_stems:
        for suffix in ['.svg', '.html', '.tiff', '.tif', '.png']:
            (supplementary_dir / f'{legacy_stem}{suffix}').unlink(missing_ok=True)

    write_manifest(
        generated_paths,
        figures_dir / 'figure_suite_manifest.json',
        input_dir=input_dir,
        figures_dir=figures_dir,
    )
    print(f'[generate_manuscript_figures_html_svg] figures_dir={figures_dir}')
    print(f'[generate_manuscript_figures_html_svg] supplementary_dir={supplementary_dir}')


if __name__ == '__main__':
    main()
