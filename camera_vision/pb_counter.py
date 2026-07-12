# -*- coding: utf-8 -*-
"""
PB 합판 장수 카운팅 시스템 v4 (로컬 실행용) - RGF 텍스처 억제 + 이방성 형태학 + 곡선 경계 추적

v4 변경 사항 (PB 표면 거친 텍스처로 인한 중앙부 오탐지 대응):
  - Rolling Guidance Filter(RGF)로 스케일 기반 텍스처 억제 (cv2.ximgproc 없이
    분리형 Joint Bilateral Filter를 직접 구현하여 재현)
  - Sobel 수평 엣지맵 + 비등방성(가로로 긴) 형태학 열림/닫힘으로 경계선만 남김
    (※ FFT 기반 동적 distance 추정은 실제 사진 검증 중 ROI에 배경이 조금만
       섞여도 오탐지가 폭증해 채택하지 않음. 고정 distance=50이 더 안정적)

[설치] (최초 1회)
    pip install opencv-python numpy scipy

[실행]
    python pb_counter_local.py
        -> 사진 경로를 생략하면 스크립트와 같은 폴더의 test.jpg를 사용
    python pb_counter_local.py 사진.jpg
        -> 창이 뜨면 마우스로 "판 더미 한 층(밴드)"을 드래그해서 선택 후
           Enter (다시 선택하려면 c)
    python pb_counter_local.py 사진.jpg --top 370 --bot 3765
        -> 밴드 상단/하단 y좌표를 직접 지정 (창 안 뜸)
    python pb_counter_local.py 사진.jpg --t 18
        -> 18T 자재 섹션일 때

[출력]
    - 콘솔: 판 수, 실재고(임시판 2장 제외), pitch
    - 사진과 같은 폴더에 결과이미지_annotated.jpg 저장 + 결과 창 표시
"""
import argparse
import os
import sys

import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = os.path.join(BASE_DIR, "test.jpg")


# ---------- Windows 한글 경로 대응 입출력 ----------
def imread_kr(path):
    """cv2.imread는 Windows에서 한글/공백 경로를 못 읽으므로 우회"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 열 수 없습니다: {path}")
    return img


def imwrite_kr(path, img):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


# ---------- RGF (Rolling Guidance Filter) : cv2.ximgproc 없이 재현 ----------
def _joint_bilateral_pass(src, guide, axis, radius, sigma_space, sigma_color):
    """1축 방향 Joint Bilateral Filter (reflect 패딩으로 경계 왜곡 방지).

    분리형(separable) 근사이므로 완전한 2D JBF와 동일하지는 않지만,
    cv2.ximgproc 없이도 스케일 인지형 텍스처 억제 효과를 재현한다.
    """
    pad = [(0, 0), (0, 0)]
    pad[axis] = (radius, radius)
    src_p = np.pad(src, pad, mode="reflect")
    guide_p = np.pad(guide, pad, mode="reflect")

    acc = np.zeros_like(src, dtype=np.float32)
    wsum = np.zeros_like(src, dtype=np.float32)
    n = src.shape[axis]
    for d in range(-radius, radius + 1):
        sl = slice(radius + d, radius + d + n)
        if axis == 1:
            src_shift, guide_shift = src_p[:, sl], guide_p[:, sl]
        else:
            src_shift, guide_shift = src_p[sl, :], guide_p[sl, :]
        spatial_w = np.exp(-(d * d) / (2.0 * sigma_space ** 2))
        range_w = np.exp(-((guide_shift - guide) ** 2) / (2.0 * sigma_color ** 2))
        w = spatial_w * range_w
        acc += w * src_shift
        wsum += w
    return acc / np.maximum(wsum, 1e-6)


def rolling_guidance_filter(gray, sigma_s=15.0, sigma_r=30.0, iterations=4):
    """PDF 4.2절 RGF: (1) 대형 가우시안 블러로 소규모 텍스처 소거 -> 초기 가이드 J1
    (2) Joint Bilateral Filter를 반복 적용해 원본의 큰 경계(합판 갭)만 복원."""
    I = gray.astype(np.float32)
    J = cv2.GaussianBlur(I, (0, 0), sigmaX=sigma_s)   # 목재 칩 반경보다 큰 sigma
    radius = max(2, int(round(sigma_s * 2)))
    for _ in range(iterations):
        J = _joint_bilateral_pass(I, J, axis=1, radius=radius,
                                   sigma_space=sigma_s, sigma_color=sigma_r)
        J = _joint_bilateral_pass(J, J, axis=0, radius=radius,
                                   sigma_space=sigma_s, sigma_color=sigma_r)
    return J


# ---------- 핵심 알고리즘 ----------
def count_band(img, y_top, y_bot, thickness_mm=23.0):
    H, W = img.shape[:2]

    # 1단계: ROI 제한 - 좌우 가장자리 톱밥/정렬불량 노이즈를 배제하고
    #         중앙 35% 폭만 사용 (PDF 8장 1단계)
    x0, x1 = int(W * 0.325), int(W * 0.675)
    roi = cv2.cvtColor(img[y_top:y_bot, x0:x1], cv2.COLOR_BGR2GRAY)
    roi = cv2.createCLAHE(2.0, (8, 8)).apply(roi)

    # 2단계: RGF로 PB 표면의 미세 목재 칩 텍스처를 스케일 기준으로 소거
    smoothed = rolling_guidance_filter(roi, sigma_s=15.0, sigma_r=30.0, iterations=4)

    # 3단계: 수평 경계 강조(Sobel-y) + 이진화 + 비등방성 형태학 열림/닫힘
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3))
    _, binary = cv2.threshold(sobel_y, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 물리적 간격은 실제로는 1~2px 두께로만 잡힐 때가 많아, 세로 3px을 요구하는
    # 열림 커널에 그대로 넣으면 진짜 경계까지 침식돼 사라진다. 먼저 세로로 살짝
    # 팽창시켜 두께를 확보한 뒤 이방성 열림/닫힘을 적용한다.
    roi_w = binary.shape[1]
    thickened = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (min(50, max(1, roi_w - 1)), 3))
    opened = cv2.morphologyEx(thickened, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (min(100, max(1, roi_w - 1)), 1))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

    # 4단계: 1D 수평 투영 프로파일에서 피크(=갭) 탐지
    #   PDF는 FFT로 합판 주기를 추정해 distance를 동적으로 산출할 것을 제안했으나,
    #   실제 사진으로 검증한 결과 ROI에 배경/여백이 조금만 섞여도 추정 주기가
    #   반토막 나거나 배로 튀어 오탐지가 폭증했다 (예: 42장 -> 68장).
    #   반면 고정 distance=50(원래 v3 알고리즘 값)은 동일 사진에서 밴드 경계를
    #   여러 번 바꿔도 41~42개로 일관되게 검출되어, 텍스처가 이미 RGF로
    #   억제된 상태에서는 고정값이 더 신뢰할 수 있는 것으로 확인되어 채택한다.
    P = gaussian_filter1d(closed.sum(axis=1).astype(float), 2)
    v, _ = find_peaks(P, distance=50, prominence=np.std(P) * 0.4, width=3)
    if len(v) < 3:
        raise RuntimeError("판 경계가 거의 검출되지 않았습니다. "
                           "밴드 선택 범위/사진 초점을 확인하세요.")
    v = (v + y_top).astype(float)

    # 누락된 경계 보간 (기존 로직 유지: 인접 간격 대비 큰 공백을 등분해 채움)
    sp = np.diff(v)
    filled, iflag = [v[0]], [False]
    for i in range(1, len(v)):
        local = np.median(sp[max(0, i - 3): min(len(sp), i + 2)])
        n_miss = int(round(sp[i - 1] / local)) - 1
        for m in range(n_miss):
            filled.append(v[i - 1] + (m + 1) * sp[i - 1] / (n_miss + 1))
            iflag.append(True)
        filled.append(v[i]); iflag.append(False)
    b, iflag = np.array(filled), np.array(iflag)
    pt = np.median(np.diff(b[:6])); pb = np.median(np.diff(b[-6:]))
    keep = (b - y_top >= 0.4 * pt) & (y_bot - b >= 0.4 * pb)
    b, iflag = b[keep], iflag[keep]

    sheets = len(b) + 1
    stock = sheets - 2          # 업무 규칙: 보강목 위아래 임시판 2장 제외
    pitch = float(np.median(np.diff(b)))

    # 5단계: 검증 - 인접 피크 간격의 변동이 크면(=오탐지/누락 의심) 경고
    cv = float(np.std(np.diff(b)) / pitch) if pitch > 0 else 0.0
    if cv > 0.3:
        print(f"[경고] 경계 간격 변동이 큽니다 (변동계수={cv:.2f}). "
              f"오탐지/누락 가능성이 있으니 결과 이미지를 확인하세요.", file=sys.stderr)

    # 시각화: 형태학 처리로 노이즈가 제거된 closed 맵에서 스트립별 국소 최대값을
    #        따라가며 곡선 라인을 그린다 (기울기/휨이 있어도 실제 틈을 추적).
    #        원본 Sobel 맵을 그대로 쓰면 잔여 텍스처 때문에 선이 심하게 흔들려서,
    #        이미 이방성 열림/닫힘을 거쳐 정제된 맵을 사용한다.
    n = 12
    edges = np.linspace(0, roi.shape[1] - 1, n + 1).astype(int)
    cols = [(edges[i] + edges[i + 1]) // 2 + x0 for i in range(n)]
    vis = img.copy()
    half = int(pitch * 0.45)
    for y, ip in zip(b, iflag):
        ys = []
        for j in range(n):
            yi = int(y - y_top)
            lo, hi = max(0, yi - half), min(closed.shape[0], yi + half)
            seg = closed[lo:hi, edges[j]:edges[j + 1]].mean(axis=1)
            ys.append(lo + int(np.argmax(seg)) + y_top if seg.max() > 0 and len(seg) > 3 else int(y))
        ys = np.clip(np.array(ys, float),
                     np.median(ys) - half, np.median(ys) + half)
        ys = gaussian_filter1d(ys, 1.0, mode="nearest")  # 스트립 간 튐 완화
        pts = np.column_stack([cols, ys]).astype(np.int32)
        color = (255, 150, 0) if ip else (0, 0, 255)   # 주황 = 보간된 경계
        cv2.polylines(vis, [pts], False, color, 3)
    cv2.rectangle(vis, (x0, y_top), (x1, y_bot), (0, 255, 0), 4)
    cv2.putText(vis, f"sheets={sheets}  stock={stock}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 2.6, (0, 0, 255), 6)
    return sheets, stock, pitch, vis


def _screen_size(default=(1920, 1080)):
    """Windows 작업 화면 해상도 조회 (실패 시 기본값)."""
    if sys.platform != "win32":
        return default
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return default


# ---------- 마우스 밴드 선택 ----------
def select_band(img, max_h=None):
    """축소 미리보기에서 밴드를 드래그 선택 -> 원본 좌표로 환산.

    창 제목은 OpenCV가 Windows에서 한글 제목을 깨뜨려 표시하는 문제를 피하려고
    영문으로 고정한다. 창은 크기 조절 가능(WINDOW_NORMAL)하게 만들어 화면
    해상도에 맞춰 최대화/드래그 확대해서 정밀하게 선택할 수 있게 한다.
    """
    if max_h is None:
        _, screen_h = _screen_size()
        max_h = int(screen_h * 0.85)
    scale = min(1.0, max_h / img.shape[0])
    prev = cv2.resize(img, None, fx=scale, fy=scale)
    print("\n[안내] 판 더미 한 층(밴드)을 마우스로 드래그하세요.")
    print("       선택 후 Enter/Space = 확정, c = 다시 선택")
    print("       (창 가장자리를 드래그하면 크기를 더 키울 수 있습니다)\n")
    win = "Select Band (drag, then Enter)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, prev.shape[1], prev.shape[0])
    r = cv2.selectROI(win, prev, showCrosshair=False, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = r
    if w == 0 or h == 0:
        sys.exit("선택이 취소되었습니다.")
    return int(y / scale), int((y + h) / scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=DEFAULT_IMAGE,
                    help=f"더미 정면 사진 경로 (생략 시 {os.path.basename(DEFAULT_IMAGE)})")
    ap.add_argument("--top", type=int, help="밴드 상단 y (생략 시 마우스 선택)")
    ap.add_argument("--bot", type=int, help="밴드 하단 y")
    ap.add_argument("--t", type=float, default=23.0, help="자재 두께 mm (기본 23)")
    ap.add_argument("--no-window", action="store_true",
                    help="결과 창을 띄우지 않음 (파일 저장만)")
    args = ap.parse_args()

    img = imread_kr(args.image)
    print(f"이미지 크기: {img.shape[1]} x {img.shape[0]}")

    if args.top is not None and args.bot is not None:
        y_top, y_bot = args.top, args.bot
    else:
        y_top, y_bot = select_band(img)
    print(f"밴드 범위: y = {y_top} ~ {y_bot}")

    sheets, stock, pitch, vis = count_band(img, y_top, y_bot, args.t)

    print("-" * 46)
    print(f"판 수      : {sheets} 장")
    print(f"실재고     : {stock} 장  (임시판 2장 제외)")
    print(f"pitch      : {pitch:.0f} px  ({args.t / pitch:.3f} mm/px)")
    print("-" * 46)

    base, _ = os.path.splitext(args.image)
    out = base + "_annotated.jpg"
    try:
        imwrite_kr(out, vis)
    except OSError:  # 사진 폴더가 읽기 전용이면 현재 폴더에 저장
        out = os.path.join(os.getcwd(),
                           os.path.basename(base) + "_annotated.jpg")
        imwrite_kr(out, vis)
    print(f"결과 이미지 저장: {out}")

    if not args.no_window:
        h = 900
        small = cv2.resize(vis, None, fx=h / vis.shape[0], fy=h / vis.shape[0])
        cv2.imshow("result (press any key)", small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()