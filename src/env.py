# 環境変数の管理を行うスクリプト.
# 複数のpythonスクリプトで同じ環境変数を使うことがあるので,
# 管理専用のスクリプトを作成した.

import os

DEVICE = "cuda"  # GPU が無い場合は "cpu"
CACHE = True  # データセットをキャッシュするかどうか

YOLO_VERSION = "26"  # 8は使えない. yolov8と書くが, yolo11, yolo26と書くため
MODEL_SIZE = "n"  # n, s, m, l, x
IMAGE_SIZE = 640
EPOCHS = 1
PACIENCE = 30
FRACTION = 0.01
BATCH = -1  # -1で自動調整, 1以上で固定
DATASET_ROOT = "dataset"

ROBOFLOW_API_KEY = None
ROBOFLOW_WORKSPACE = None
ROBOFLOW_PRETRAINED_PROJECT = None
ROBOFLOW_TEST_PROJECT = None
ROBOFLOW_COLOR_PROJECT = None


def read_env():
    """環境変数から設定を読み込む。"""
    global ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PRETRAINED_PROJECT, ROBOFLOW_TEST_PROJECT, ROBOFLOW_COLOR_PROJECT
    ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]
    ROBOFLOW_WORKSPACE = os.environ["ROBOFLOW_WORKSPACE"]
    ROBOFLOW_PRETRAINED_PROJECT = os.environ["ROBOFLOW_PRETRAINED_PROJECT"]
    ROBOFLOW_TEST_PROJECT = os.environ["ROBOFLOW_TEST_PROJECT"]
    ROBOFLOW_COLOR_PROJECT = os.environ["ROBOFLOW_COLOR_PROJECT"]

    global DEVICE, CACHE, YOLO_VERSION, MODEL_SIZE, IMAGE_SIZE, EPOCHS, PACIENCE, FRACTION, BATCH, DATASET_ROOT
    DEVICE = os.environ.get("DEVICE", DEVICE)
    CACHE = os.environ.get("CACHE", CACHE)
    YOLO_VERSION = os.environ.get("YOLO_VERSION", YOLO_VERSION)
    MODEL_SIZE = os.environ.get("MODEL_SIZE", MODEL_SIZE)
    IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", IMAGE_SIZE))
    EPOCHS = int(os.environ.get("EPOCHS", EPOCHS))
    PACIENCE = int(os.environ.get("PACIENCE", PACIENCE))
    FRACTION = float(os.environ.get("FRACTION", FRACTION))
    BATCH = int(os.environ.get("BATCH", BATCH))
    DATASET_ROOT = os.environ.get("DATASET_ROOT", DATASET_ROOT)


read_env()
