from ultralytics import YOLO
from roboflow import Roboflow
import os
import glob
from pathlib import Path

TEST_IMAGE_DIR = "./data/pokerchips"
DEVICE = "cuda"  # GPU が無い場合は "cpu"

YOLO_VERSION = "26"
EPOCHS = 300
FRACTION = 0.04

ROBOFLOW_API_KEY = "example"
ROBOFLOW_WORKSPACE = "example@gmail.com"
ROBOFLOW_PROJECT = "example-project"


def collect_images(directory, extensions):
    """指定フォルダから対象拡張子の画像パスを集める（大文字小文字どちらも）。"""
    files = []
    for ext in extensions:
        files += glob.glob(os.path.join(directory, f"*.{ext}"))
        files += glob.glob(os.path.join(directory, f"*.{ext.upper()}"))
    return sorted(set(files))


def download_dataset():
    """Roboflow からデータを YOLOv{YOLO_VERSION} 形式でダウンロードする。"""
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(1)
    return version.download(f"yolov{YOLO_VERSION}")


def train_model(dataset, epochs, fraction):
    """学習済み YOLO に追加学習を行い、結果を返す。"""
    model = YOLO(f"yolov{YOLO_VERSION}-seg.pt")
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=epochs,
        fraction=fraction,
        imgsz=640,
        device=DEVICE,
        name="chip-segment",
        amp=True,
        batch=-1,
    )
    return (model, results)


def detect(weights_path, image_dir):
    """学習済みモデルでテスト画像を検出する。結果画像は save=True で自動保存。"""
    images = collect_images(image_dir, extensions=("jpg", "jpeg", "png"))
    if not images:
        print("テスト画像が見つかりませんでした。")
        return None

    model = YOLO(weights_path)
    results = model(
        images,  # リストごと渡せる
        save=True,
        show_boxes=False,
        show_labels=False,
        show_conf=False,
        line_width=2,
    )

    save_dir = results[0].save_dir
    print(f"検出結果の画像を保存しました: {save_dir}")
    for f in collect_images(save_dir, extensions=("jpg", "jpeg", "png")):
        print(" -", f)
    return results


def report_artifacts(save_dir):
    """学習過程の各種プロットが保存されているか確認し、パスを案内する。"""
    d = Path(save_dir)
    want = [
        "val_batch0_labels.jpg",  # 自分のアノテーション
        "val_batch0_pred.jpg",  # 同じ画像への予測
        "results.png",  # 学習曲線
        "MaskPR_curve.png",  # P-R トレードオフ
        "MaskF1_curve.png",  # conf 閾値の目安
        "confusion_matrix_normalized.png",  # 混同行列
        "labels.jpg",  # データ分布
        "train_batch0.jpg",  # 拡張後の入力
    ]
    print("=" * 60)
    print("学習過程の出力ファイル:")
    for name in want:
        p = d / name
        print(f"  {'○' if p.exists() else '×(なし)'}  {name} -> {p}")


def read_env():
    """環境変数から設定を読み込む。"""
    global ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT
    ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", ROBOFLOW_API_KEY)
    ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", ROBOFLOW_WORKSPACE)
    ROBOFLOW_PROJECT = os.environ.get("ROBOFLOW_PROJECT", ROBOFLOW_PROJECT)


def main():
    read_env()
    dataset = download_dataset()
    model, results = train_model(dataset, epochs=EPOCHS, fraction=FRACTION)
    detect(model.trainer.best, TEST_IMAGE_DIR)
    report_artifacts(results.save_dir)


if __name__ == "__main__":
    main()
