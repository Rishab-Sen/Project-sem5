# -*- coding: utf-8 -*-
"""Render Bootcamp pages to PNG + build a contact sheet HTML with WhatsApp images."""
import os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pymupdf

DOCS = r"C:\Users\Rishab Sen\Documents"
OUT = ".freebuff/render"
os.makedirs(OUT, exist_ok=True)

doc = pymupdf.open(os.path.join(DOCS, "5 Days Cybersecurity Bootcamp.pdf"))
for pno in range(len(doc)):
    pix = doc[pno].get_pixmap(dpi=100)
    pix.save(f"{OUT}/bootcamp_p{pno+1}.png")
doc.close()
print("bootcamp pages rendered")

# copy whatsapp images (resize to keep small)
from PIL import Image
import glob
thumbs = []
for i, p in enumerate(glob.glob(os.path.join(DOCS, "WhatsApp Image*.jpeg"))):
    im = Image.open(p).convert("RGB")
    im.thumbnail((480, 640))
    fn = f"{OUT}/wa_{i}.jpg"
    im.save(fn, quality=80)
    thumbs.append((os.path.basename(p), fn, im.size))
    print("thumb:", os.path.basename(p), "->", fn, im.size)

# also thumbnail the logo
logo = os.path.join(OUT, "logo.jpg")
im = Image.open(os.path.join(DOCS, ".freebuff_imgs_check") if False else ".freebuff/imgs/B_p1_x33.jpeg").convert("RGB")
im.save(logo, quality=90)

html = ["<html><body style='font-family:Arial;background:#222;color:#fff'>"]
html.append("<h2>Bootcamp format reference</h2>")
for n in [1, 2, 7]:
    html.append(f"<div style='margin:8px'><img src='bootcamp_p{n}.png' style='width:420px;border:1px solid #888'></div>")
html.append("<h2>WhatsApp images (possible workshop photos)</h2><table><tr>")
for i, (name, fn, size) in enumerate(thumbs):
    html.append(f"<td style='padding:6px'><img src='wa_{i}.jpg' style='width:200px'><br><small>{name}</small></td>")
    if (i+1) % 3 == 0:
        html.append("</tr><tr>")
html.append("</tr></table>")
html.append("<h2>Logo</h2><img src='logo.jpg' style='width:200px;background:#fff'>")
html.append("</body></html>")
with open(f"{OUT}/sheet.html", "w", encoding="utf-8") as f:
    f.write("\n".join(html))
print("sheet written")
