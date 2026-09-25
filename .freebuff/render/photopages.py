# -*- coding: utf-8 -*-
import base64, sys, pymupdf
doc = pymupdf.open(r"C:\Users\Rishab Sen\Documents\Workshop_Report.pdf")
zoom = 2.2
pages = [6, 7]
imgs = []
for p in pages:
    pix = doc[p].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    imgs.append(base64.b64encode(pix.tobytes("jpg", jpg_quality=82)).decode())
html = ('<!doctype html><html><body style="margin:0;background:#333;display:flex;gap:8px;padding:8px">'
        + ''.join(f'<img src="data:image/jpeg;base64,{b}" style="width:49%">'
                  for b in imgs) + '</body></html>')
with open(r"C:\Users\Rishab Sen\Desktop\Project\.freebuff\render\photopages.html", "w") as f:
    f.write(html)
print("written")
