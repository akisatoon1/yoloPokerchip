from ultralytics import YOLO
from os import path

import env
from training import evaluate, show_predictions
import utils


def finetune(name, weights_path, yaml_path, freeze):
    model = YOLO(weights_path)
    model.train(
        data=yaml_path,
        epochs=env.EPOCHS,
        patience=env.PACIENCE,
        fraction=env.FRACTION,
        imgsz=env.IMAGE_SIZE,
        device=env.DEVICE,
        name=name,
        amp=True,
        batch=-1,
        cache=env.CACHE,
        freeze=freeze,
        optimizer="AdamW",
        lr0=0.001,
    )
    return model


def fintune_and_evaluate(freeze):
    test_dataset = utils.download_dataset(
        env.ROBOFLOW_TEST_PROJECT,
        version_n=4,
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
