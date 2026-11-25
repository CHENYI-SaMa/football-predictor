# -*- coding: utf-8 -*-
"""生成应用图标：蓝色圆底 + 白色足球五边形。"""
from PIL import Image, ImageDraw

S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# 圆底
d.ellipse([8, 8, S - 8, S - 8], fill=(0, 122, 255, 255))
# 内圈白环
d.ellipse([40, 40, S - 40, S - 40], outline=(255, 255, 255, 255), width=6)
# 中心五边形（足球标志）
cx, cy, R = S / 2, S / 2, 62
import math
pts = []
for i in range(5):
    ang = -math.pi / 2 + i * 2 * math.pi / 5
    pts.append((cx + R * math.cos(ang), cy + R * math.sin(ang)))
d.polygon(pts, fill=(255, 255, 255, 255))
# 五条放射线
for p in pts:
    d.line([(cx, cy), p], fill=(255, 255, 255, 255), width=8)

# 多种尺寸 ico
img.save(r"C:\Users\EDY\WorkBuddy\2026-08-07-15-37-36\football-prediction\app_icon.png")
sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
img.save(r"C:\Users\EDY\WorkBuddy\2026-08-07-15-37-36\football-prediction\app_icon.ico",
         sizes=sizes)
print("图标已生成: app_icon.ico")
