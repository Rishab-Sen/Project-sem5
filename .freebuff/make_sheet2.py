# -*- coding: utf-8 -*-
"""Render output PDF pages to PNG and build a self-contained data-URI contact sheet."""
import os, io, sys, base64
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pymupdf

OUT_DIR = ".freebuff/render"
os.makedirs(OUT_DIR, exist_ok=True)

pdf_path = os.path.join(r"C:\Users\Rishab Sen\Documents", "Workshop_Report_Formatted.pdf")
doc = pymupdf.open(pdf_path)
print("pages:", len(doc))
parts = ["<html><body style='background:#333;font-family:Arial;color:#fff'>",
         "<h2>Workshop_Report_Formatted.pdf</h2>",
         "<div style='display:flex;flex-wrap:wrap;gap:10px'>"]
for pno in range(len(doc)):
    pix = doc[pno].get_pixmap(dpi=90)
    fn = f"{OUT_DIR}/out_p{pno+1}.png"
    pix.save(fn)
    b64 = base64.b64encode(open(fn, "rb").read()).decode()
    parts.append(f"<div><img src='data:image/png;base64,{b64}' style='width:360px;border:2px solid #eee'>"
                 f"<div style='text-align:center'>page {pno+1}</div></div>")
parts.append("</div>")

# reference pages too
ref = pymupdf.open(os.path.join(r"C:\Users\Rishab Sen\Documents", "5 Days Cybersecurity Bootcamp.pdf"))
parts.append("<h2>Reference: 5 Days Cybersecurity Bootcamp.pdf</h2><div style='display:flex;flex-wrap:wrap;gap:10px'>")
for pno in [0, 1, 6]:
    pix = ref[pno].get_pixmap(dpi=90)
    b64 = base64.b64encode(pix.tobytes("png")).decode()
    parts.append(f"<div><img src='data:image/png;base64,{b64}' style='width:360px;border:2px solid #eee'>"
                 f"<div style='text-align:center'>ref page {pno+1}</div></div>")
parts.append("</div></body></html>")
with open(f"{OUT_DIR}/sheet2.html", "w", encoding="utf-8") as f:
    f.write("".join(parts))
print("sheet2 written, size:", os.path.getsize(f"{OUT_DIR}/sheet2.html") // 1024, "KB")
