"""
make_pdf.py — Build submission-ready PDFs from the Markdown documents
=====================================================================
Usage:  python src/make_pdf.py

Converts:
    REPORT.md             -> REPORT.pdf   (figures from plots/ embedded)
    PROJECT_EXPLAINED.md  -> PROJECT_EXPLAINED.pdf
    README.md             -> README.pdf

Pipeline: Markdown -> HTML (python-markdown) -> ReportLab flowables via
BeautifulSoup traversal. Pure-Python, no browser involved. Figures under
plots/ referenced in the Markdown are embedded as images.
"""

from __future__ import annotations

import os
import re
import sys

import markdown
from bs4 import BeautifulSoup, NavigableString, Tag

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(ROOT, "plots")

# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
INK = colors.HexColor("#16324f")
ACCENT = colors.HexColor("#2c6e49")
MUTED = colors.HexColor("#475569")
LINE = colors.HexColor("#cbd5e1")

S = {
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17,
                         leading=21, textColor=INK, spaceBefore=2,
                         spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13,
                         leading=16, textColor=INK, spaceBefore=14,
                         spaceAfter=6),
    "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=11,
                         leading=14, textColor=colors.HexColor("#2c3e50"),
                         spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.6,
                           leading=13.8, textColor=colors.HexColor("#111827"),
                           spaceAfter=5, alignment=4),  # justify
    "li": ParagraphStyle("li", parent=__import__("reportlab").platypus
                         .ParagraphStyle if False else ParagraphStyle(
                             "li", fontName="Helvetica", fontSize=9.6,
                             leading=13.5, spaceAfter=2.5),
                         leftIndent=14, bulletIndent=4, alignment=4),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=8.2,
                           leading=11.2, textColor=colors.HexColor("#e2e8f0"),
                           backColor=colors.HexColor("#0f172a"),
                           borderPadding=6, leftIndent=2, rightIndent=2,
                           spaceBefore=4, spaceAfter=6),
    "inline": ParagraphStyle("inline", parent=ParagraphStyle(
        "ip", fontName="Helvetica", fontSize=9.6, leading=13.8)),
    "quote": ParagraphStyle("quote", fontName="Helvetica-Oblique",
                            fontSize=9.4, leading=13, textColor=MUTED,
                            leftIndent=10, spaceBefore=4, spaceAfter=6,
                            borderColor=LINE, borderWidth=0.5, borderPadding=5),
    "caption": ParagraphStyle("caption", fontName="Helvetica-Oblique",
                              fontSize=8.4, leading=11, textColor=MUTED,
                              alignment=1, spaceBefore=2, spaceAfter=8),
}

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm


def esc(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def rich_text(node: Tag) -> str:
    """Convert inline HTML to ReportLab paragraph markup."""
    parts = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(esc(str(child)))
        elif child.name in ("strong", "b"):
            parts.append(f"<b>{rich_text(child)}</b>")
        elif child.name in ("em", "i"):
            parts.append(f"<i>{rich_text(child)}</i>")
        elif child.name == "code":
            parts.append(
                f"<font face='Courier' backColor='#eef2f7' size='8.4'>"
                f"{esc(child.get_text())}</font>")
        elif child.name == "a":
            parts.append(f"<u><font color='#1d4ed8'>{rich_text(child)}</font>"
                         f"</u>")
        elif child.name == "br":
            parts.append("<br/>")
        else:
            parts.append(rich_text(child))
    return "".join(parts)


def find_image(name: str) -> str | None:
    for root in (PLOTS, ROOT):
        p = os.path.join(root, name)
        if os.path.exists(p):
            return p
    return None


def image_flowable(src: str, max_w: float):
    try:
        ir = ImageReader(src)
        iw, ih = ir.getSize()
    except Exception:
        return None
    scale = min(max_w / iw, 1.0)
    return Image(src, width=iw * scale, height=ih * scale)


# --------------------------------------------------------------------------- #
# Block-level HTML -> flowables
# --------------------------------------------------------------------------- #
def render_block(node: Tag, out: list):
    name = node.name
    if name in ("h1", "h2", "h3", "h4"):
        out.append(Paragraph(rich_text(node), S["h2" if name == "h2" else
                                        "h3" if name in ("h3", "h4")
                                        else "h1"]))
    elif name == "p":
        txt = rich_text(node)
        # embed images referenced as plots/<name>.png inside paragraphs
        m = re.search(r"plots/([\w.-]+\.png)", txt)
        if m and m.group(1) in txt:
            img = find_image(m.group(1))
            if img:
                fl = image_flowable(img, PAGE_W - 2 * MARGIN - 10)
                if fl:
                    out.extend([Spacer(1, 4), fl,
                                Paragraph(m.group(1), S["caption"])])
                    return
        out.append(Paragraph(txt, S["body"]))
    elif name in ("ul", "ol"):
        for i, li in enumerate(node.find_all("li", recursive=False), 1):
            bullet = f"{i}." if name == "ol" else "\u2022"
            out.append(Paragraph(rich_text(li), S["li"], bulletText=bullet))
        out.append(Spacer(1, 3))
    elif name == "pre":
        code = node.get_text()
        lines = code.rstrip("\n").split("\n")
        # keep code blocks unbreakable when short, break long ones by chunking
        block = Paragraph("<br/>".join(esc(l) for l in lines)
                          .replace(" ", "&nbsp;"), S["code"])
        out.append(block)
    elif name == "blockquote":
        out.append(Paragraph(rich_text(node), S["quote"]))
    elif name == "table":
        header = [th.get_text(strip=True) for th in
                  node.select("thead th")] or \
                 [th.get_text(strip=True) for th in
                  node.find_all("th")]
        rows = []
        for tr in node.select("tbody tr") or node.select("tr"):
            tds = tr.find_all(["td", "th"])
            if tds and header:
                rows.append([rich_text(td) for td in tds])
        if not rows:
            return
        # first row may be header if no thead
        if not header:
            header = rows.pop(0)
        data = [[Paragraph(f"<b>{esc(h)}</b>", S["inline"]) for h in header]]
        data += [[Paragraph(c, S["inline"]) for c in r] for r in rows]
        ncols = len(header)
        avail = PAGE_W - 2 * MARGIN
        col_w = [avail / ncols] * ncols
        t = Table(data, colWidths=col_w, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.4, LINE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        out.extend([Spacer(1, 4), t, Spacer(1, 6)])
    elif name == "hr":
        out.append(Spacer(1, 6))
    else:
        # generic container: recurse
        for child in node.children:
            if isinstance(child, Tag):
                render_block(child, out)


def md_to_flowables(md_path: str) -> list:
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    html = markdown.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists"])
    soup = BeautifulSoup(html, "html.parser")
    out: list = []
    for node in soup.children:
        if isinstance(node, Tag):
            render_block(node, out)
    return out


# --------------------------------------------------------------------------- #
# PDF assembly
# --------------------------------------------------------------------------- #
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 10 * mm,
                      "MARL for Cooperative Payload Transport & Collision "
                      "Avoidance — COEP Technological University")
    canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 12 * mm, PAGE_W - MARGIN, 12 * mm)
    canvas.restoreState()


def figures_appendix() -> list:
    """Appendix with every PNG in plots/ embedded (used for REPORT.pdf)."""
    out: list = [PageBreak(), Paragraph("Appendix B — Figures", S["h1"]),
                 Paragraph("All generated by src/plots.py and "
                           "src/trajectories.py from the training runs and "
                           "formal benchmarks in runs/.", S["body"])]
    captions = {
        "learning_curves.png": "Fig. 1 — Training reward (left) and greedy "
            "full-task evaluation success (right) for MAPPO (top) and QMIX "
            "(bottom).",
        "curriculum_effect.png": "Fig. 2 — Curriculum ablation: greedy "
            "success with staged difficulty vs training from scratch.",
        "comparison_bars.png": "Fig. 3 — Final 30-episode benchmark across "
            "policies on the full task.",
        "trajectories.png": "Fig. 4 — Successful delivery traces: payload "
            "path (black), rover paths (coloured), goal (star).",
        "demo_render_test.png": "Fig. 5 — Rendered environment frame from "
            "the PyGame demo (LiDAR rays, rig lines, goal zone).",
    }
    n = 0
    for name in sorted(os.listdir(PLOTS)) if os.path.isdir(PLOTS) else []:
        if not name.lower().endswith(".png"):
            continue
        img = image_flowable(os.path.join(PLOTS, name),
                             PAGE_W - 2 * MARGIN - 10)
        if img is None:
            continue
        n += 1
        out.extend([Spacer(1, 6), img,
                    Paragraph(captions.get(name, name), S["caption"])])
    if n == 0:
        return []
    return out


def build(md_path: str, pdf_path: str, with_figures: bool = False):
    story = md_to_flowables(md_path)
    if with_figures:
        story += figures_appendix()
    doc = BaseDocTemplate(pdf_path, pagesize=A4,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=15 * mm, bottomMargin=16 * mm,
                          title=os.path.basename(pdf_path))
    frame = Frame(MARGIN, 16 * mm, PAGE_W - 2 * MARGIN,
                  PAGE_H - 31 * mm, id="main")
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame],
                                       onPage=footer)])
    doc.build(story)
    print(f"{os.path.basename(md_path)} -> {os.path.basename(pdf_path)} "
          f"({os.path.getsize(pdf_path)//1024} KB)")


def main():
    docs = ["REPORT.md", "PROJECT_EXPLAINED.md", "README.md"]
    for doc in docs:
        src = os.path.join(ROOT, doc)
        if os.path.exists(src):
            build(src, src.replace(".md", ".pdf"),
                  with_figures=(doc == "REPORT.md"))
        else:
            print(f"skip {doc} (missing)")
    print("done")


if __name__ == "__main__":
    main()
