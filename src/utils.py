from roboflow import Roboflow
from os.path import join

from env import ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, YOLO_VERSION, DATASET_ROOT


def download_dataset(project_name, version_n):
    """Roboflow からデータを YOLOv{YOLO_VERSION} 形式でダウンロードする。"""
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(project_name)
    version = project.version(version_n)
    location = join(DATASET_ROOT, f"{project_name}-v{version_n}")
    return version.download(f"yolo{YOLO_VERSION}", location=location)
