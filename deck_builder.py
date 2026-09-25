# -*- coding: utf-8 -*-
"""Build 5 COEP-template decks (revised): verified refs, diagrams, stats, notes, no overlaps."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

TEMPLATE = r"C:\Users\Rishab Sen\Documents\coep_template.pptx"
OUT_DIR = r"C:\Users\Rishab Sen\Documents"

ACCENT = RGBColor(0x0F, 0x6F, 0xC6)   # theme accent1
DARK   = RGBColor(0x17, 0x40, 0x6D)   # theme dk2
LIGHT  = RGBColor(0xDB, 0xEF, 0xF9)   # theme lt2
GRAY   = RGBColor(0x59, 0x59, 0x59)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

FOOTER_LINES = ["Department of Mechanical Engineering",
                "COEP Technological University, Pune"]

TITLE_IX = 0
BODY_IX = 1
BODY_TOP, BODY_H = 1.60, 3.50


# ---------------------------------------------------------------- primitives
def _solid(shape, rgb):
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb


def _runs(p, size, color, bold=False, italic=False, font="Calibri"):
    for r in p.runs:
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.bold = bold
        r.font.italic = italic
        r.font.name = font
        return r


def add_textbox(slide, l, t, w, h, lines, size=12, color=DARK, bold=False,
                align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, wrap=True,
                space_after=3, italic=False, font="Calibri"):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_top = tf.margin_bottom = Pt(1)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = align
        p.space_after = Pt(space_after)
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.bold = bold
            r.font.italic = italic
            r.font.name = font
    return tb


def add_footer(slide, size=10):
    add_textbox(slide, 2.63, 5.24, 4.74, 0.76, FOOTER_LINES, size=size,
                color=GRAY, align=PP_ALIGN.CENTER)


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def add_pipeline(slide, stages, top=1.66, height=0.80):
    """Adjacent chevrons sharing vertices: width split of total span, no gaps/overlaps."""
    n = len(stages)
    left, total_w = 0.42, 9.16
    w = Inches(total_w) // n
    x0 = Inches(left)
    for i, st in enumerate(stages):
        shp = slide.shapes.add_shape(MSO_SHAPE.CHEVRON, x0 + i * w, Inches(top),
                                     w, Inches(height))
        _solid(shp, ACCENT if i % 2 == 0 else DARK)
        shp.line.color.rgb = WHITE
        shp.line.width = Pt(1)
        tf = shp.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Pt(2)
        tf.margin_top = tf.margin_bottom = Pt(1)
        tf.text = st
        for p in tf.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(10)
                r.font.bold = True
                r.font.color.rgb = WHITE
                r.font.name = "Calibri"
    return x0 + n * w  # right edge (EMU)


def set_bullets(body, items, sizes=(17, 15)):
    tf = body.text_frame
    tf.clear()
    tf.word_wrap = True
    first = True
    for it in items:
        lvl, lead, rest = it
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = lvl
        p.space_after = Pt(8 if lvl == 0 else 5)
        if lead:
            r1 = p.add_run()
            r1.text = lead
            r1.font.bold = True
            r1.font.color.rgb = DARK
            r1.font.size = Pt(sizes[lvl])
            r1.font.name = "Calibri"
        if rest:
            r2 = p.add_run()
            r2.text = (" " if lead else "") + rest
            r2.font.size = Pt(sizes[lvl])
            r2.font.color.rgb = RGBColor(0x21, 0x21, 0x21)
            r2.font.name = "Calibri"


def content_slide(prs, title, bullets, notes=None, pipeline=None,
                  body_top=BODY_TOP, body_h=BODY_H, sizes=(17, 15)):
    s = prs.slides.add_slide(prs.slide_masters[0].slide_layouts[1])
    t = s.shapes.title
    t.text = title
    for p in t.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(26)
    if pipeline:
        add_pipeline(s, pipeline)
        body_top, body_h = 2.66, 2.52
        sizes = (14.5, 13)
    body = s.placeholders[1]
    body.top, body.height = Inches(body_top), Inches(body_h)
    set_bullets(body, bullets, sizes=sizes)
    add_footer(s)
    if notes:
        add_notes(s, notes)
    return s


def ref_slide(prs, title, refs, keywords=None, notes=None):
    """References slide with numbered, verified citations in a light panel."""
    s = prs.slides.add_slide(prs.slide_masters[0].slide_layouts[1])
    t = s.shapes.title
    t.text = title
    for p in t.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(26)
    body = s.placeholders[1]
    body.top, body.height = Inches(1.60), Inches(3.50)
    tf = body.text_frame
    tf.clear()
    tf.word_wrap = True
    first = True
    for i, ref in enumerate(refs, 1):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        r1 = p.add_run()
        r1.text = f"[{i}]  "
        r1.font.bold = True
        r1.font.color.rgb = ACCENT
        r1.font.size = Pt(13.5)
        r1.font.name = "Calibri"
        r2 = p.add_run()
        r2.text = ref
        r2.font.size = Pt(13.5)
        r2.font.color.rgb = RGBColor(0x21, 0x21, 0x21)
        r2.font.name = "Calibri"
        p.space_after = Pt(10)
    if keywords:
        p = tf.add_paragraph()
        r = p.add_run()
        r.text = "Keywords: "
        r.font.bold = True
        r.font.color.rgb = DARK
        r.font.size = Pt(13.5)
        r.font.name = "Calibri"
        r2 = p.add_run()
        r2.text = keywords
        r2.font.size = Pt(13.5)
        r2.font.color.rgb = RGBColor(0x21, 0x21, 0x21)
        r2.font.name = "Calibri"
    add_footer(s)
    if notes:
        add_notes(s, notes)
    return s


def stat_card(slide, x, y, w, h, value, label, fill=ACCENT, vsize=26, lsize=10.5):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y),
                                 Inches(w), Inches(h))
    _solid(box, fill)
    box.line.fill.background()
    try:
        box.adjustments[0] = 0.10
    except Exception:
        pass
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_top = tf.margin_bottom = Pt(3)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = value
    p.alignment = PP_ALIGN.CENTER
    for r in p.runs:
        r.font.size = Pt(vsize)
        r.font.bold = True
        r.font.color.rgb = WHITE
        r.font.name = "Calibri"
    p2 = tf.add_paragraph()
    p2.text = label
    p2.alignment = PP_ALIGN.CENTER
    for r in p2.runs:
        r.font.size = Pt(lsize)
        r.font.color.rgb = WHITE
        r.font.name = "Calibri"
    return box


# ------------------------------------------------------------ slide builders
def stat_band_slide(prs, title, stats, bullets, notes=None):
    """Slide with a row of big-number stat cards on top, bullets below."""
    s = prs.slides.add_slide(prs.slide_masters[0].slide_layouts[1])
    t = s.shapes.title
    t.text = title
    for p in t.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(26)
    n = len(stats)
    gap = 0.15
    w = (9.2 - gap * (n - 1)) / n
    x = 0.40
    for value, label, fill in stats:
        stat_card(s, x, 1.62, w, 1.02, value, label, fill=fill)
        x += w + gap
    body = s.placeholders[1]
    body.top, body.height = Inches(2.90), Inches(2.30)
    set_bullets(body, bullets, sizes=(14.5, 13))
    add_footer(s)
    if notes:
        add_notes(s, notes)
    return s


def architecture_slide(prs, title, pipeline, blocks, notes=None, block_top=2.72):
    """Pipeline chevrons + row of labelled architecture blocks below."""
    s = prs.slides.add_slide(prs.slide_masters[0].slide_layouts[1])
    t = s.shapes.title
    t.text = title
    for p in t.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(26)
    add_pipeline(s, pipeline)
    n = len(blocks)
    gap = 0.14
    w = (9.16 - gap * (n - 1)) / n
    x = 0.42
    for head, desc, fill in blocks:
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),
                                 Inches(block_top), Inches(w), Inches(1.18))
        _solid(box, fill)
        box.line.fill.background()
        try:
            box.adjustments[0] = 0.07
        except Exception:
            pass
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Pt(4)
        tf.margin_top = tf.margin_bottom = Pt(3)
        p = tf.paragraphs[0]
        p.text = head
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.size = Pt(11.5)
            r.font.bold = True
            r.font.color.rgb = WHITE
            r.font.name = "Calibri"
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.alignment = PP_ALIGN.CENTER
        for r in p2.runs:
            r.font.size = Pt(9.5)
            r.font.color.rgb = WHITE
            r.font.name = "Calibri"
        x += w + gap
    add_footer(s)
    if notes:
        add_notes(s, notes)
    return s


def title_slide(prs, title, subtitle_lines, notes=None):
    s = prs.slides.add_slide(prs.slide_masters[0].slide_layouts[0])
    s.shapes.title.text = title
    sub = s.placeholders[1]
    tf = sub.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, line in enumerate(subtitle_lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.alignment = PP_ALIGN.CENTER
        p.space_after = Pt(4)
        for r in p.runs:
            r.font.size = Pt(15)
            r.font.color.rgb = DARK
            r.font.name = "Calibri"
    add_footer(s, size=11)
    if notes:
        add_notes(s, notes)
    return s


def thankyou_slide(prs, deck_note, notes=None):
    s = prs.slides.add_slide(prs.slide_masters[0].slide_layouts[2])
    s.shapes.title.text = "Thank You"
    for p in s.shapes.title.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(40)
    try:
        sub = s.placeholders[1]
        tf = sub.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.text = "Questions & Discussion"
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.size = Pt(18)
            r.font.color.rgb = DARK
        p2 = tf.add_paragraph()
        p2.text = deck_note
        p2.alignment = PP_ALIGN.CENTER
        for r in p2.runs:
            r.font.size = Pt(13)
            r.font.color.rgb = GRAY
    except (KeyError, IndexError):
        pass
    add_footer(s)
    if notes:
        add_notes(s, notes)
    return s


def build(fname, topic_title, slides, deck_note, title_subs=None, title_note=None,
          thanks_note=None):
    prs = Presentation(TEMPLATE)
    sldIdLst = prs.slides._sldIdLst
    sld = list(sldIdLst)[0]
    prs.part.drop_rel(sld.rId)
    sldIdLst.remove(sld)
    cp = prs.core_properties
    cp.title = topic_title
    cp.author = "Department of Mechanical Engineering, COEP Technological University"

    subs = title_subs or ["AI-ML Semester Project Proposal",
                          "Robotics & Artificial Intelligence",
                          "Presented by: ____________________     Guide: Dr. S. S. Ohol"]
    subs = subs + ["Course: Robotics & AI, COEP Technological University"]
    title_slide(prs, topic_title, subs, notes=title_note)

    for sl in slides:
        notes = sl.get("notes")
        kind = sl.get("kind", "content")
        if kind == "stats":
            stat_band_slide(prs, sl["title"], sl["stats"], sl["bullets"], notes=notes)
        elif kind == "arch":
            architecture_slide(prs, sl["title"], sl["pipeline"], sl["blocks"], notes=notes)
        elif kind == "ref":
            ref_slide(prs, sl["title"], sl["refs"], sl.get("keywords"), notes=notes)
        else:
            content_slide(prs, sl["title"], sl["bullets"], notes=notes,
                          pipeline=sl.get("pipeline"))

    thankyou_slide(prs, deck_note, notes=thanks_note)
    out = OUT_DIR + "\\" + fname
    prs.save(out)
    print("Saved:", out, f"({len(prs.slides._sldIdLst)} slides)")
    return out
