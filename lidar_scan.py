# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(BASE_DIR, "raw-sensor-data-office.csv")

JUMP_TH  = 0.15      # 거리 불연속 기준 15cm -> 15cm 이상 거리가 갑자기 변하면 벽으로 인식
LINE_TH  = 0.05      # 직선에서 5cm 벗어나면 벽
MIN_PTS  = 2        # edge 사이 점이 2개 이하면 noise 취급해서 버림

# ---------- 1~3. 파싱 / 필터 / 정렬 / 변환  ----------
angles, dists, quals = [], [], []
with open(RAW_FILE) as f:
    for line in f:
        line = line.strip()
        if "\t" not in line: continue
        parts = line.split(",b'")[0].split("\t") # 시리얼 통신 데이터 제거, 4등분
        if len(parts) != 4: continue            # 4등분 안되면 ㅌㅌ
        angles.append(float(parts[2])); dists.append(float(parts[3])); quals.append(int(parts[1]))
angles = np.array(angles); dists = np.array(dists) / 1000.0; quals = np.array(quals)

valid = (dists > 0.1) & (quals >= 10)
a_ok, d_ok = angles[valid], dists[valid]
order = np.argsort(a_ok); a_ok, d_ok = a_ok[order], d_ok[order]
th = np.deg2rad(a_ok)
x, y = d_ok * np.cos(th), d_ok * np.sin(th)

# ---------- 4.(거리 불연속) ----------
dd = np.abs(np.diff(d_ok))
edge_idx = np.where(dd > JUMP_TH)[0]
seg_bounds = np.concatenate(([0], edge_idx + 1, [len(d_ok)]))
raw_segments = [(s, e) for s, e in zip(seg_bounds[:-1], seg_bounds[1:]) if e - s >= MIN_PTS]

# ---------- 5. (방향 불연속 = 코너) ----------
def point_to_chord_dist(xs, ys):
    """구간의 모든 포인트에서 '양 끝점을 이은 직선(현)'까지의 수직거리[m] 배열"""
    x1, y1, x2, y2 = xs[0], ys[0], xs[-1], ys[-1]
    L = np.hypot(x2 - x1, y2 - y1)
    if L < 1e-9:
        return np.zeros(len(xs))
    # 공식: |(x2-x1)(y1-yi) - (x1-xi)(y2-y1)| / L
    return np.abs((x2 - x1) * (y1 - ys) - (x1 - xs) * (y2 - y1)) / L

def split_straight(s, e, corners):
    """[s,e) 구간이 직선이 아니면 최대 이탈 지점(코너)에서 재귀 분할"""
    if e - s < MIN_PTS:
        return []
    dev = point_to_chord_dist(x[s:e], y[s:e])
    k = int(np.argmax(dev))
    if dev[k] <= LINE_TH:
        return [(s, e)]                        # 전체가 직선
    corners.append(s + k)                      # 코너 위치 기록
    return split_straight(s, s + k + 1, corners) + split_straight(s + k, e, corners)

walls, corners = [], []
for s, e in raw_segments:
    walls += split_straight(s, e, corners)

# ---------- 6. 벽 길이 계산 및 출력 ----------
result = []
for s, e in walls:
    p1, p2 = (x[s], y[s]), (x[e-1], y[e-1])
    span = float(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
    result.append((s, e, p1, p2, span))
result.sort(key=lambda t: -t[4])

print(f"1차(jump edge) 군집 {len(raw_segments)}개 -> 코너 {len(corners)}개 발견 "
      f"-> 최종 직선 벽 {len(result)}개\n")
print(" ID | 포인트수 | 길이[m] | 직선성(최대 이탈 cm)")
print("-" * 50)
for i, (s, e, p1, p2, span) in enumerate(result, 1):
    max_dev = point_to_chord_dist(x[s:e], y[s:e]).max() * 100
    print(f" {i:2d} |  {e-s:4d}   | {span:.3f}  |  {max_dev:.1f}")

# ---------- 7. 시각화 ----------
fig, ax = plt.subplots(figsize=(9.5, 8))
ax.scatter(x, y, s=3, c="lightgray")
colors = plt.cm.tab20.colors
for i, (s, e, p1, p2, span) in enumerate(result):
    ax.scatter(x[s:e], y[s:e], s=4, color=colors[i % 20])
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", color=colors[i % 20], lw=0.8)
    if span > 0.15:
        ax.annotate(f"{span:.2f}m", ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2), fontsize=7)
ax.scatter(x[corners], y[corners], s=70, facecolors="none",
           edgecolors="green", lw=1.5, label=f"corners found ({len(corners)})")
ax.plot(0, 0, "k*", ms=14, label="LIDAR")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("Cloud point and distance")
ax.set_aspect("equal"); ax.legend(loc="lower left")
plt.tight_layout()
SAVE_FILE = os.path.join(BASE_DIR, "lidar_distance.png")
plt.savefig(SAVE_FILE, dpi=110)
print("\n그래프 저장: lidar_distance.png")
