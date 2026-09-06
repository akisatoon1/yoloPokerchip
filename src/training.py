from ultralytics import YOLO
from roboflow import Roboflow
import os

from show_pred import save_pred_imgs
from count_chips import save_counts_from_result
import utils
import env


def train_model(dataset, epochs, patience, fraction):
    """学習済み YOLO に追加学習を行い、結果を返す。"""
    model = YOLO(f"yolo{env.YOLO_VERSION}{env.MODEL_SIZE}-seg.pt")
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=epochs,
        patience=patience,
        fraction=fraction,
        imgsz=env.IMAGE_SIZE,
        device=env.DEVICE,
        name="training-online-dataset",
        amp=True,
        batch=env.BATCH,
        cache=env.CACHE,
        mosaic=0.0,
    )
    return model


# 毎回bestを読み込むのは無駄なため
BEST_MODEL = None


def evaluate(dataset_dir, name, split="test", model=None):
    """dataset の test(default) セットを使って学習済みモデルを評価する。"""
    if model is None:
        model = BEST_MODEL
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


def show_predictions(imgs_dir, name, model=BEST_MODEL, conf=0.25, iou=0.7):
    """学習済みモデルの推論を行い、結果を保存する。

    デフォルトの推論結果の画像がわかりづらいため。
    """
    if model is None:
        model = BEST_MODEL

    results = model.predict(
        source=imgs_dir,
        imgsz=env.IMAGE_SIZE,
        device=env.DEVICE,
        name=name,
        save_txt=True,
        save_conf=True,
        conf=conf,
        iou=iou,
    )
    save_pred_imgs(results)


def main():
    pretrained_dataset = utils.download_dataset(
        env.ROBOFLOW_PRETRAINED_PROJECT,
        version_n=1,
    )
    model = train_model(
        pretrained_dataset,
        epochs=env.EPOCHS,
        patience=env.PACIENCE,
        fraction=env.FRACTION,
    )
    test_dataset = utils.download_dataset(
        env.ROBOFLOW_TEST_PROJECT,
        version_n=5,
    )

    global BEST_MODEL
    BEST_MODEL = YOLO(model.trainer.best)

    if env.DEVICE == "cuda":
        # cpuのときはなぜかここでKilledされる
        evaluate(pretrained_dataset.location, name="val-pretrained", split="val")
    evaluate(test_dataset.location, name="val-mytask", split="val")
    show_predictions(
        os.path.join(pretrained_dataset.location, "valid", "images"),
        name="show-val-predictions",
    )
    show_predictions(
        os.path.join(test_dataset.location, "valid", "images"),
        name="show-mytask-predictions",
    )


if __name__ == "__main__":
    main()
