# 環境変数の管理を行うスクリプト.
# 複数のpythonスクリプトで同じ環境変数を使うことがあるので,
# 管理専用のスクリプトを作成した.

import os

DEVICE = None  # GPU が無い場合は "cpu"
CACHE = None  # データセットをキャッシュするかどうか

YOLO_MODEL = None  # yolo26n-seg.pt など
IMAGE_SIZE = None
EPOCHS = None
PACIENCE = None
FRACTION = None
BATCH = None  # -1で自動調整, 1以上で固定
DATASET_ROOT = None

ROBOFLOW_API_KEY = None
ROBOFLOW_WORKSPACE = None
ROBOFLOW_PRETRAINED_PROJECT = None
ROBOFLOW_TEST_PROJECT = None
ROBOFLOW_COLOR_PROJECT = None
ROBOFLOW_BOX_PROJECT = None


def read_env():
    """環境変数から設定を読み込む。"""

    try:
        global ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PRETRAINED_PROJECT, ROBOFLOW_TEST_PROJECT, ROBOFLOW_COLOR_PROJECT, ROBOFLOW_BOX_PROJECT
        ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]
        ROBOFLOW_WORKSPACE = os.environ["ROBOFLOW_WORKSPACE"]
        ROBOFLOW_PRETRAINED_PROJECT = os.environ["ROBOFLOW_PRETRAINED_PROJECT"]
        ROBOFLOW_TEST_PROJECT = os.environ["ROBOFLOW_TEST_PROJECT"]
        ROBOFLOW_COLOR_PROJECT = os.environ["ROBOFLOW_COLOR_PROJECT"]
        ROBOFLOW_BOX_PROJECT = os.environ["ROBOFLOW_BOX_PROJECT"]

        global DEVICE, CACHE, YOLO_MODEL, IMAGE_SIZE, EPOCHS, PACIENCE, FRACTION, BATCH, DATASET_ROOT
        DEVICE = os.environ["DEVICE"]
        CACHE = os.environ["CACHE"]
        YOLO_MODEL = os.environ["YOLO_MODEL"]
        IMAGE_SIZE = int(os.environ["IMAGE_SIZE"])
        EPOCHS = int(os.environ["EPOCHS"])
        PACIENCE = int(os.environ["PACIENCE"])
        FRACTION = float(os.environ["FRACTION"])
        BATCH = int(os.environ["BATCH"])
        DATASET_ROOT = os.environ["DATASET_ROOT"]
    except KeyError as e:
        raise KeyError(f"環境変数 {e} が設定されていません。") from e


read_env()
