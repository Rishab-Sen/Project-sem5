# -*- coding: utf-8 -*-
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pymupdf

DOCS = r"C:\Users\Rishab Sen\Documents"

# ---------- 1. Full text dump of Workshop_Report.pdf ----------
doc = pymupdf.open(os.path.join(DOCS, "Workshop_Report.pdf"))
with open(".freebuff/workshop_full_text.txt", "w", encoding="utf-8") as f:
    for pno, page in enumerate(doc):
        f.write(f"\n\n===== PAGE {pno+1} =====\n")
        f.write(page.get_text("text"))
doc.close()
print("workshop_full_text.txt written")

# ---------- 2. Bootcamp PDF styling inspection ----------
doc = pymupdf.open(os.path.join(DOCS, "5 Days Cybersecurity Bootcamp.pdf"))

print("\n\n########## BOOTCAMP STYLING ##########")
for pno in [0, 1, 6]:  # page 1 (title+logo), page 2 (body), page 7 (prepared-by)
    page = doc[pno]
    print(f"\n===== PAGE {pno+1} =====")
    print("--- TEXT SPANS (font/size/color/origin) ---")
    d = page.get_text("dict")
    for block in d["blocks"]:
        if block["type"] != 0:
            print(f"  [IMAGE BLOCK] bbox={tuple(round(v,1) for v in block['bbox'])}")
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                color = span["color"]
                r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
                txt = span["text"].strip()
                if not txt:
                    continue
                print(f"  font={span['font']:<28} size={span['size']:>5.1f} rgb=({r},{g},{b}) "
                      f"bbox_x0={span['bbox'][0]:.0f} y0={span['bbox'][1]:.0f} | {txt[:70]}")
    print("--- IMAGES (placement) ---")
    for img in page.get_images(full=True):
        xref = img[0]
        rects = page.get_image_rects(xref)
        info = doc.extract_image(xref)
        for rr in rects:
            print(f"  xref={xref} placed at bbox=({rr.x0:.0f},{rr.y0:.0f},{rr.x1:.0f},{rr.y1:.0f}) "
                  f"native={info['width']}x{info['height']} {info['ext']}")
    print("--- VECTOR DRAWINGS (first 12) ---")
    for i, drw in enumerate(page.get_drawings()):
        if i >= 12:
            print("  ... more")
            break
        items = [it[0] for it in drw["items"]][:6]
        fill = drw.get("fill")
        fill_s = tuple(round(c, 2) for c in fill) if fill else None
        stroke = drw.get("color")
        stroke_s = tuple(round(c, 2) for c in stroke) if stroke else None
        print(f"  rect=({drw['rect'].x0:.0f},{drw['rect'].y0:.0f},{drw['rect'].x1:.0f},{drw['rect'].y1:.0f}) "
              f"fill={fill_s} stroke={stroke_s} width={drw.get('width')} items={items}")
doc.close()
print("\nDONE")
