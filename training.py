from ultralytics import YOLO
from roboflow import Roboflow
import os
import glob
from show_pred import save_pred_imgs
from count_chips import save_counts_from_result

DEVICE = "cuda"  # GPU が無い場合は "cpu"

YOLO_VERSION = "26"  # 8は使えない. yolov8と書くが, yolo11, yolo26と書くため
EPOCHS = 1
PACIENCE = 30
FRACTION = 0.01
DATASET_ROOT = "dataset"

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

    global DEVICE, YOLO_VERSION, EPOCHS, PACIENCE, FRACTION, DATASET_ROOT
    DEVICE = os.environ.get("DEVICE", DEVICE)
    YOLO_VERSION = os.environ.get("YOLO_VERSION", YOLO_VERSION)
    EPOCHS = int(os.environ.get("EPOCHS", EPOCHS))
    PACIENCE = int(os.environ.get("PACIENCE", PACIENCE))
    FRACTION = float(os.environ.get("FRACTION", FRACTION))
    DATASET_ROOT = os.environ.get("DATASET_ROOT", DATASET_ROOT)


def collect_images(directory, extensions):
    """指定フォルダから対象拡張子の画像パスを集める（大文字小文字どちらも）。"""
    files = []
    for ext in extensions:
        files += glob.glob(os.path.join(directory, f"*.{ext}"))
        files += glob.glob(os.path.join(directory, f"*.{ext.upper()}"))
    return sorted(set(files))


def download_dataset(api_key, workspace, project_name):
    """Roboflow からデータを YOLOv{YOLO_VERSION} 形式でダウンロードする。"""
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(1)
    location = os.path.join(DATASET_ROOT, project_name)
    return version.download(f"yolo{YOLO_VERSION}", location=location)


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
        name="training-online-dataset",
        amp=True,
        batch=-1,
    )
    return model


# 毎回bestを読み込むのは無駄なため
BEST_MODEL = None


def evaluate(dataset, name, split="test", model=None):
    """dataset の test(default) セットを使って学習済みモデルを評価する。"""
    if model is None:
        model = BEST_MODEL

    results = model.val(
        data=f"{dataset.location}/data.yaml",
        split=split,
        imgsz=640,
        device=DEVICE,
        name=name,
        save_txt=True,
        save_conf=True,
    )

    # 現状のデータセットでは, splitがvalのときデータはvalid/に保存されるため
    split_dir = "valid" if split == "val" else split
    save_counts_from_result(results, dataset, split_dir=split_dir)


def show_predictions(source, name, model=BEST_MODEL, conf=0.25, iou=0.7):
    """学習済みモデルの推論を行い、結果を保存する。

    デフォルトの推論結果の画像がわかりづらいため。
    """
    if model is None:
        model = BEST_MODEL

    results = model.predict(
        source=source,
        imgsz=640,
        device=DEVICE,
        name=name,
        save_txt=True,
        save_conf=True,
        conf=conf,
        iou=iou,
    )
    save_pred_imgs(results)


def main():
    pretrained_dataset = download_dataset(
        ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PRETRAINED_PROJECT
    )
    model = train_model(
        pretrained_dataset, epochs=EPOCHS, patience=PACIENCE, fraction=FRACTION
    )
    test_dataset = download_dataset(
        ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_TEST_PROJECT
    )

    global BEST_MODEL
    BEST_MODEL = YOLO(model.trainer.best)

    if DEVICE == "cuda":
        # cpuのときはなぜかここでKilledされる
        evaluate(pretrained_dataset, name="val-pretrained", split="val")
    evaluate(test_dataset, name="val-mytask")
    show_predictions(
        os.path.join(pretrained_dataset.location, "valid", "images"),
        name="show-val-predictions",
    )
    show_predictions(
        os.path.join(test_dataset.location, "test", "images"),
        name="show-mytask-predictions",
    )


read_env()
if __name__ == "__main__":
    main()
