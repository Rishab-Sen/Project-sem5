# -*- coding: utf-8 -*-
"""Embed grid.png as base64 into grid.html so the preview can display it."""
import base64, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
b64 = base64.b64encode(open("render/grid.png", "rb").read()).decode()
html = ("<!doctype html><html><body style='margin:0;background:#222'>"
        f"<img src='data:image/png;base64,{b64}' style='width:100%'></body></html>")
open("render/grid.html", "w", encoding="utf-8").write(html)
print("grid.html written,", len(b64) // 1024, "KB b64")
