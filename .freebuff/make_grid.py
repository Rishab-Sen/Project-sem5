# -*- coding: utf-8 -*-
"""Combine output page PNGs into one compact grid image."""
import os, io, sys, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from PIL import Image, ImageDraw

pages = sorted(glob.glob("render/out_p*.png"), key=lambda p: int(p.split("out_p")[1].split(".")[0]))
cols, tw = 4, 330
ims = []
for p in pages:
    im = Image.open(p)
    im.thumbnail((tw, 10000))
    ims.append(im)
rows = (len(ims) + cols - 1) // cols
rh = max(i.height for i in ims) + 24
sheet = Image.new("RGB", (cols * (tw + 10) + 10, rows * rh + 10), (40, 40, 40))
d = ImageDraw.Draw(sheet)
for i, im in enumerate(ims):
    x = 10 + (i % cols) * (tw + 10)
    y = 10 + (i // cols) * rh
    sheet.paste(im, (x, y + 18))
    d.text((x, y + 2), f"page {i+1}", fill=(255, 255, 0))
sheet.save("render/grid.png")
print("grid saved", sheet.size)
