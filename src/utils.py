from os.path import join

from roboflow import Roboflow

import env
from src.count_chips import save_counts_from_result


def download_dataset(project_name, version_n):
    """Roboflow からデータを YOLOv{YOLO_VERSION} 形式でダウンロードする。"""
    rf = Roboflow(api_key=env.ROBOFLOW_API_KEY)
    project = rf.workspace(env.ROBOFLOW_WORKSPACE).project(project_name)
    version = project.version(version_n)
    location = join(env.DATASET_ROOT, f"{project_name}-v{version_n}")
    return version.download(f"yolo{env.YOLO_VERSION}", location=location)


def evaluate(name, model, dataset_dir, split="test"):
    """dataset の test(default) セットを使って学習済みモデルを評価する。"""
    results = model.val(
        data=f"{dataset_dir}/data.yaml",
        split=split,
        imgsz=env.IMAGE_SIZE,
        device=env.DEVICE,
        name=name,
        save_txt=True,
        save_conf=True,
    )

    # 現状のデータセットでは, splitがvalのときデータはvalid/に保存されるため
    split_dir = "valid" if split == "val" else split
    save_counts_from_result(results, dataset_dir, split_dir=split_dir)
