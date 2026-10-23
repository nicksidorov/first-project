"""Render the XeMat brief from structured, plain-text report blocks.

This is presentation only: no statistical results are calculated or changed.
Run ``python render_pdf.py`` beside content.json and its images, or pass --root.
"""
from __future__ import annotations

import argparse
from importlib.util import find_spec
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, LongTable, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, TableStyle,
)

# Matplotlib bundles the same fonts on Windows as well as Linux/macOS.
# Finding its data directory without importing it avoids creating a font cache.
_matplotlib = find_spec('matplotlib')
FONT_DIR = (Path(_matplotlib.origin).parent / 'mpl-data' / 'fonts' / 'ttf'
            if _matplotlib and _matplotlib.origin else Path('/usr/share/fonts/truetype/dejavu'))
INK = colors.HexColor('#192b3d')
BLUE = colors.HexColor('#17617d')
MUTED = colors.HexColor('#536372')


def register_fonts(font_dir: Path) -> None:
    """Embed real Cyrillic-capable TrueType fonts in the exported PDF."""
    for suffix, file_name in [
        ('', 'DejaVuSans.ttf'),
        ('-Bold', 'DejaVuSans-Bold.ttf'),
        ('-Oblique', 'DejaVuSans-Oblique.ttf'),
        ('-BoldOblique', 'DejaVuSans-BoldOblique.ttf'),
    ]:
        path = font_dir / file_name
        if not path.is_file():
            raise FileNotFoundError(f'Missing embedded font: {path}; use --font-dir')
        pdfmetrics.registerFont(TTFont('XeMatSans' + suffix, str(path)))
    pdfmetrics.registerFontFamily('XeMatSans', normal='XeMatSans',
        bold='XeMatSans-Bold', italic='XeMatSans-Oblique',
        boldItalic='XeMatSans-BoldOblique')


def plain(value) -> str:
    """Input is plain text; never interpret scientific angle signs as markup."""
    return escape('—' if value is None else str(value)).replace('\n', '<br/>')


def styles():
    base = getSampleStyleSheet()
    return {
        'title': ParagraphStyle('XeMatTitle', parent=base['Title'],
            fontName='XeMatSans-Bold', fontSize=17, leading=21,
            textColor=INK, alignment=TA_LEFT, spaceAfter=13, keepWithNext=True),
        'heading': ParagraphStyle('XeMatHeading', parent=base['Heading2'],
            fontName='XeMatSans-Bold', fontSize=12.2, leading=15.3,
            textColor=BLUE, spaceBefore=9, spaceAfter=6, keepWithNext=True),
        'paragraph': ParagraphStyle('XeMatParagraph', parent=base['BodyText'],
            fontName='XeMatSans', fontSize=9.7, leading=13,
            textColor=INK, spaceAfter=6, allowWidows=0, allowOrphans=0),
        'caption': ParagraphStyle('XeMatCaption', parent=base['BodyText'],
            fontName='XeMatSans', fontSize=8.5, leading=11.4,
            textColor=MUTED, spaceBefore=4, spaceAfter=10,
            allowWidows=0, allowOrphans=0),
        'cell': ParagraphStyle('XeMatCell', parent=base['BodyText'],
            fontName='XeMatSans', fontSize=8.6, leading=11.1,
            textColor=INK, spaceAfter=0, spaceBefore=0),
        'header': ParagraphStyle('XeMatTableHeader', parent=base['BodyText'],
            fontName='XeMatSans-Bold', fontSize=8.3, leading=10.8,
            textColor=colors.white, spaceAfter=0, spaceBefore=0),
    }


def make_table(block, available_width, style_map):
    headers = block['headers']
    rows = block['rows']
    if not headers or any(len(row) != len(headers) for row in rows):
        raise ValueError('Each table row must have the same length as headers')
    proportions = block.get('widths', [1] * len(headers))
    if len(proportions) != len(headers) or any(w <= 0 for w in proportions):
        raise ValueError('Table widths must be positive, one for each column')
    widths = [available_width * weight / sum(proportions) for weight in proportions]
    cells = [[Paragraph(plain(value), style_map['header']) for value in headers]]
    cells.extend([Paragraph(plain(value), style_map['cell']) for value in row] for row in rows)
    table = LongTable(cells, colWidths=widths, repeatRows=1,
                      hAlign='LEFT', splitByRow=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
            [colors.HexColor('#f1f6f8'), colors.white]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, BLUE),
        ('LINEBELOW', (0, -1), (-1, -1), 0.4, colors.HexColor('#ccd8df')),
    ]))
    return table


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#d1dce3'))
    canvas.setLineWidth(.35)
    canvas.line(doc.leftMargin, 14 * mm, A4[0] - doc.rightMargin, 14 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont('XeMatSans', 7.5)
    canvas.drawString(doc.leftMargin, 9.5 * mm, 'XeMat · 22 мая · Детектор 221 · Проверка на 100 000 пробах')
    canvas.drawRightString(A4[0] - doc.rightMargin, 9.5 * mm, str(doc.page))
    canvas.restoreState()


def render(root: Path, output: Path | None = None, font_dir: Path = FONT_DIR) -> Path:
    root = root.resolve()
    content = json.loads((root / 'content.json').read_text(encoding='utf-8'))
    register_fonts(font_dir)
    style_map = styles()
    output = output or root / 'report_ru.pdf'
    doc = SimpleDocTemplate(str(output), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=17 * mm, bottomMargin=20 * mm,
        title=content['title'], author='XeMat',
        subject='Lc, ложные обнаружения и номинальный MDC: чистая проба, 22 мая, 100000 проверочных реализаций',
        pageCompression=1)
    story = [Paragraph(plain(content['title']), style_map['title'])]
    section = []

    def flush_section():
        # Keep a logical section on one page whenever it fits a full page.
        # Larger sections remain splittable by ReportLab, including tables.
        if section:
            story.append(KeepTogether(section[:]))
            section.clear()

    for block in content['blocks']:
        kind = block['type']
        if kind == 'heading':
            flush_section()
        if kind in ('heading', 'paragraph'):
            if block.get('text', '').strip():
                section.append(Paragraph(plain(block['text']), style_map[kind]))
            else:
                section.append(Spacer(1, 5))
        elif kind == 'table':
            section.append(make_table(block, doc.width, style_map))
            section.append(Spacer(1, 8))
        elif kind == 'image':
            image_path = (root / block['file']).resolve()
            image_path.relative_to(root)
            image_width, image_height = ImageReader(str(image_path)).getSize()
            scale = min(doc.width / image_width, 120 * mm / image_height)
            figure = Image(str(image_path), width=image_width * scale,
                height=image_height * scale, hAlign='CENTER')
            figure.keepWithNext = bool(block.get('caption'))
            section.append(figure)
            if block.get('caption'):
                section.append(Paragraph(plain(block['caption']), style_map['caption']))
            else:
                section.append(Spacer(1, 8))
        elif kind == 'page_break':
            flush_section()
            story.append(PageBreak())
        else:
            raise ValueError(f'Unsupported content block type: {kind}')
    flush_section()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--font-dir', type=Path, default=FONT_DIR)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = render(args.root, args.output, args.font_dir)
    print(f'PDF written: {result} ({result.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
