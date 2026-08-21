from ultralytics import YOLO
from roboflow import Roboflow
import os
import glob
from pathlib import Path

DEVICE = "cuda"  # GPU が無い場合は "cpu"

YOLO_VERSION = "26"  # 8は使えない. yolov8と書くが, yolo11, yolo26と書くため
EPOCHS = 300
PACIENCE = 30
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

    global DEVICE, YOLO_VERSION, EPOCHS, PACIENCE, FRACTION
    DEVICE = os.environ.get("DEVICE", DEVICE)
    YOLO_VERSION = os.environ.get("YOLO_VERSION", YOLO_VERSION)
    EPOCHS = int(os.environ.get("EPOCHS", EPOCHS))
    PACIENCE = int(os.environ.get("PACIENCE", PACIENCE))
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


def train_model(dataset, epochs, patience, fraction):
    """学習済み YOLO に追加学習を行い、結果を返す。"""
    model = YOLO(f"yolo{YOLO_VERSION}n-seg.pt")
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=epochs,
        patience=patience,
        fraction=fraction,
        imgsz=640,
        device=DEVICE,
        name="chip-segment",
        amp=True,
        batch=-1,
    )
    return model


def evaluate(weights_path, dataset):
    """dataset の test セットを使って学習済みモデルを評価する。"""
    model = YOLO(weights_path)
    model.val(
        data=f"{dataset.location}/data.yaml",
        split="test",
        imgsz=640,
        device=DEVICE,
        name="test-chip-segment",
        batch=-1,
    )


def main():
    read_env()
    pretrained_dataset = download_dataset(
        ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PRETRAINED_PROJECT
    )
    model = train_model(
        pretrained_dataset, epochs=EPOCHS, patience=PACIENCE, fraction=FRACTION
    )
    test_dataset = download_dataset(
        ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_TEST_PROJECT
    )
    evaluate(model.trainer.best, test_dataset)


if __name__ == "__main__":
    main()
