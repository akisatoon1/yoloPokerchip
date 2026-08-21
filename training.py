from ultralytics import YOLO
from roboflow import Roboflow
import os
import glob
from pathlib import Path

TEST_IMAGE_DIR = "./data/pokerchips"
DEVICE = "cuda"  # GPU が無い場合は "cpu"

YOLO_VERSION = "26"  # 8は使えない. yolov8と書くが, yolo11, yolo26と書くため
EPOCHS = 300
FRACTION = 0.04

ROBOFLOW_API_KEY = None
ROBOFLOW_WORKSPACE = None
ROBOFLOW_PRETRAINED_PROJECT = None
ROBOFLOW_TEST_PROJECT = None


def read_env():
    """環境変数から設定を読み込む。"""
    global ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PRETRAINED_PROJECT, ROBOFLOW_TEST_PROJECT
    ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]
    ROBOFLOW_WORKSPACE = os.environ["ROBOFLOW_WORKSPACE"]
    ROBOFLOW_PRETRAINED_PROJECT = os.environ["ROBOFLOW_PRETRAINED_PROJECT"]
    ROBOFLOW_TEST_PROJECT = os.environ["ROBOFLOW_TEST_PROJECT"]

    global TEST_IMAGE_DIR, DEVICE, YOLO_VERSION, EPOCHS, FRACTION
    TEST_IMAGE_DIR = os.environ.get("TEST_IMAGE_DIR", TEST_IMAGE_DIR)
    DEVICE = os.environ.get("DEVICE", DEVICE)
    YOLO_VERSION = os.environ.get("YOLO_VERSION", YOLO_VERSION)
    EPOCHS = int(os.environ.get("EPOCHS", EPOCHS))
    FRACTION = float(os.environ.get("FRACTION", FRACTION))


def collect_images(directory, extensions):
    """指定フォルダから対象拡張子の画像パスを集める（大文字小文字どちらも）。"""
    files = []
    for ext in extensions:
        files += glob.glob(os.path.join(directory, f"*.{ext}"))
        files += glob.glob(os.path.join(directory, f"*.{ext.upper()}"))
    return sorted(set(files))


def download_dataset(api_key, workspace, project):
    """Roboflow からデータを YOLOv{YOLO_VERSION} 形式でダウンロードする。"""
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    version = project.version(1)
    return version.download(f"yolo{YOLO_VERSION}")


def train_model(dataset, epochs, fraction):
    """学習済み YOLO に追加学習を行い、結果を返す。"""
    model = YOLO(f"yolo{YOLO_VERSION}n-seg.pt")
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=epochs,
        fraction=fraction,
        imgsz=640,
        device=DEVICE,
        name="chip-segment",
        amp=True,
        batch=-1,
        patience=30,
    )
    return model


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


def main():
    read_env()
    dataset = download_dataset(
        ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PRETRAINED_PROJECT
    )
    model = train_model(dataset, epochs=EPOCHS, fraction=FRACTION)
    detect(model.trainer.best, TEST_IMAGE_DIR)


if __name__ == "__main__":
    main()
