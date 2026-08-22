"""
インスタンスセグメンテーションのtxtを読み、インスタンスごとに別色で画面表示する関数。
バウンディングボックスは描かない。

conf_thres の扱い（1つの引数でモードを切り替える）:
  - None (デフォルト): 末尾に信頼度が「無い」txt（アノテーション） → 全インスタンスを描画
  - 数値 (例 0.25):   末尾が信頼度の txt（予測 / save_conf=True） → conf未満を足切り
"""

import cv2
import colorsys
import numpy as np
import matplotlib.pyplot as plt

"""
requirements に追加すべし: opencv-python, matplotlib, numpy
"""


def show_instance_seg(
    img, label_txt, conf_thres=None, alpha=0.45, thickness=2, fill=True
):
    """
    Args:
        img:        画像パス(str) または BGR の ndarray
        label_txt:  ラベルtxtのパス。
                    - conf_thres=None のとき  : class x1 y1 ... xn yn        （信頼度なし）
                    - conf_thres=数値 のとき  : class x1 y1 ... xn yn conf   （末尾が信頼度）
                    座標はいずれも 0〜1 正規化。
        conf_thres: None なら足切りしない（末尾に信頼度が無い前提）。
                    数値を渡すとその値未満のインスタンスを描画しない（末尾が信頼度の前提）。
        alpha:      塗りの濃さ。fill=Falseなら無視される
        thickness:  輪郭線の太さ
        fill:       Trueで半透明塗り+輪郭、Falseで輪郭線のみ（重なりが激しいとき有効）

    Returns:
        描画済みの BGR ndarray（保存などに使いたい場合に受け取れる）
    """
    has_conf = conf_thres is not None  # ★ ここでモードが決まる

    # --- 画像読み込み ---
    image = cv2.imread(img) if isinstance(img, str) else img.copy()
    if image is None:
        raise FileNotFoundError(f"画像を読み込めません: {img}")
    h, w = image.shape[:2]

    # --- txtをパース ---
    min_tokens = 8 if has_conf else 7  # class(1)+最低3点(6)[+conf(1)]
    instances = []
    with open(label_txt) as f:
        for line in f:
            p = line.split()
            if len(p) < min_tokens:
                continue
            cls = int(float(p[0]))
            if has_conf:
                conf = float(p[-1])
                if conf < conf_thres:  # ★ 信頼度で足切り
                    continue
                coord_tokens = p[1:-1]  # 末尾(conf)を除いた座標
            else:
                conf = None
                coord_tokens = p[1:]  # 座標のみ（信頼度なし）
            xy = np.array(list(map(float, coord_tokens)), dtype=np.float32).reshape(
                -1, 2
            )
            xy[:, 0] *= w  # 正規化 → ピクセル座標
            xy[:, 1] *= h
            instances.append((cls, conf, xy.astype(np.int32)))

    # --- インスタンスごとの色（黄金比で色相を回して隣接色を離す, BGR） ---
    colors = []
    for i in range(max(len(instances), 1)):
        hue = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.70, 1.0)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))

    # --- 描画（box は描かない） ---
    out = image.copy()
    if fill:
        overlay = image.copy()
        for (cls, conf, pts), c in zip(instances, colors):
            cv2.fillPoly(overlay, [pts], c)
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    for (cls, conf, pts), c in zip(instances, colors):
        cv2.polylines(out, [pts], isClosed=True, color=c, thickness=thickness)

    # --- matplotlib で表示（OpenCVのBGR → RGB に変換） ---
    subtitle = f"conf >= {conf_thres}" if has_conf else "all (no conf filter)"
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.title(f"instances: {len(instances)}  ({subtitle})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return out


if __name__ == "__main__":
    # (1) アノテーション（信頼度なし）: conf_thres は付けない = None
    show_instance_seg(
        img="chip-count-my-env-1/test/images/IMG_5567_jpg.rf.5206ebd4fc0358f39e8295b697ae019f.jpg",
        label_txt="chip-count-my-env-1/test/labels/IMG_5567_jpg.rf.5206ebd4fc0358f39e8295b697ae019f.txt",
    )

    # (2) 予測（信頼度あり / save_conf=True）: conf_thres に数値を渡す
    # show_instance_seg(
    #     img="chip-count-my-env-1/test/images/IMG_5543....jpg",
    #     label_txt="runs/segment/predict/labels/IMG_5543....txt",
    #     conf_thres=0.25,
    # )
