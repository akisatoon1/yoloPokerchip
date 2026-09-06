import os

from ultralytics import YOLO

import env
import utils


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
        utils.evaluate(
            name="val-pretrained",
            model=BEST_MODEL,
            dataset_dir=pretrained_dataset.location,
            split="val",
        )
    utils.evaluate(
        name="val-mytask",
        model=BEST_MODEL,
        dataset_dir=test_dataset.location,
        split="val",
    )
    utils.show_predictions(
        name="show-val-predictions",
        model=BEST_MODEL,
        imgs_dir=os.path.join(pretrained_dataset.location, "valid", "images"),
    )
    utils.show_predictions(
        name="show-mytask-predictions",
        model=BEST_MODEL,
        imgs_dir=os.path.join(test_dataset.location, "valid", "images"),
    )


if __name__ == "__main__":
    main()
