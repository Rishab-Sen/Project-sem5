# -*- coding: utf-8 -*-
import os, io, sys, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from PIL import Image, ImageFilter, ImageStat

DOCS = r"C:\Users\Rishab Sen\Documents"
for p in sorted(glob.glob(os.path.join(DOCS, "WhatsApp Image*.jpeg"))):
    im = Image.open(p).convert("RGB")
    g = im.convert("L")
    sat = ImageStat.Stat(im).stddev
    edges = g.filter(ImageFilter.FIND_EDGES)
    ed = ImageStat.Stat(edges).mean[0]
    # mean brightness of background (corner patch)
    corner = g.crop((0, 0, 80, 80))
    bright = ImageStat.Stat(corner).mean[0]
    print(f"{os.path.basename(p):<50} sat=({sat[0]:5.1f},{sat[1]:5.1f},{sat[2]:5.1f}) edge={ed:5.1f} corner_brightness={bright:5.1f}")
