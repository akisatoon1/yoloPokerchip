import os


def count_lines_from_txt(path, conf_thres=None):
    """txt の有効行数を数える。conf_thres 指定時は末尾(信頼度)>=thres の行のみ。"""
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path) as f:
        for line in f:
            p = line.split()
            if conf_thres is None:
                if (
                    len(p) >= 7
                ):  # class + 最低3点(アノテーション)の最低7つ要素が必要だから
                    n += 1
            else:
                if len(p) >= 8 and float(p[-1]) >= conf_thres:  # 予測: 信頼度で足切り
                    n += 1
    return n


if __name__ == "__main__":
    print(
        count_lines_from_txt(
            "chip-count-my-env-1/test/labels/IMG_5552_jpg.rf.00a70b8ec97155031985af0447e98da4.txt"
        )
    )
