from ultralytics import YOLO
from os import path

from training import (
    CACHE,
    EPOCHS,
    PACIENCE,
    FRACTION,
    DEVICE,
    IMAGE_SIZE,
    ROBOFLOW_API_KEY,
    ROBOFLOW_WORKSPACE,
    ROBOFLOW_TEST_PROJECT,
)
from training import download_dataset, evaluate, show_predictions


def finetune(name, weights_path, yaml_path, freeze):
    model = YOLO(weights_path)
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        patience=PACIENCE,
        fraction=FRACTION,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        name=name,
        amp=True,
        batch=-1,
        cache=CACHE,
        freeze=freeze,
        optimizer="AdamW",
        lr0=0.001,
    )
    return model


def fintune_and_evaluate(freeze):
    test_dataset = download_dataset(
        ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_TEST_PROJECT, version_n=4
    )

    w_path = "pastData/sizes/training2-for-imgsz-1280/training-online-dataset-2/weights/best.pt"
    model = finetune(
        f"finetune-mytask-freeze-{freeze}",
        w_path,
        path.join(test_dataset.location, "data.yaml"),
        freeze,
    )

    evaluate(
        test_dataset.location,
        f"val-finetuned-freeze-{freeze}",
        split="val",
        model=model,
    )
    show_predictions(
        path.join(test_dataset.location, "valid", "images"),
        f"show-finetuned-preds-{freeze}",
        model=model,
    )


def main():
    freezes = [23]
    for freeze in freezes:
        fintune_and_evaluate(freeze)


if __name__ == "__main__":
    main()
