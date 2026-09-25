# -*- coding: utf-8 -*-
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pymupdf

DOCS = r"C:\Users\Rishab Sen\Documents"
OUT = os.path.join(".freebuff", "imgs")
os.makedirs(OUT, exist_ok=True)

for name in ["Workshop_Report.pdf", "5 Days Cybersecurity Bootcamp.pdf"]:
    path = os.path.join(DOCS, name)
    doc = pymupdf.open(path)
    print("=" * 80)
    print(f"FILE: {name}  |  pages: {len(doc)}  |  metadata: {doc.metadata.get('title')!r} {doc.metadata.get('author')!r}")
    tag = "W" if name.startswith("Workshop") else "B"
    for pno, page in enumerate(doc):
        r = page.rect
        print(f"--- page {pno+1}: {r.width:.1f}x{r.height:.1f}")
        txt = page.get_text("text")
        # compress whitespace for readability
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        for l in lines[:40]:
            print(f"    | {l[:110]}")
        if len(lines) > 40:
            print(f"    ... ({len(lines)-40} more lines)")
        # image inventory
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                info = doc.extract_image(xref)
                fn = f"{OUT}/{tag}_p{pno+1}_x{xref}.{info['ext']}"
                with open(fn, "wb") as f:
                    f.write(info["image"])
                print(f"    [IMG] xref={xref} {info['width']}x{info['height']} {info['ext']} -> {fn}")
            except Exception as e:
                print(f"    [IMG] xref={xref} extract failed: {e}")
        # drawings summary (rects/fills) to understand decorations
        dr = page.get_drawings()
        if dr:
            fills = {}
            for d in dr:
                if d.get("fill"):
                    c = tuple(round(v, 2) for v in d["fill"])
                    fills[c] = fills.get(c, 0) + 1
            print(f"    [DRAW] {len(dr)} drawing items; fill colors: {fills}")
    doc.close()
print("DONE")
