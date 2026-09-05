from training import (
    download_dataset,
    train_model,
    evaluate,
    show_predictions,
    read_env,
    ROBOFLOW_API_KEY,
    ROBOFLOW_WORKSPACE,
    ROBOFLOW_COLOR_PROJECT,
    EPOCHS,
    PACIENCE,
    FRACTION,
)
from ultralytics import YOLO
from os.path import join


def main():
    dataset = download_dataset(
        ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_COLOR_PROJECT, version_n=2
    )
    model = train_model(dataset, epochs=EPOCHS, patience=PACIENCE, fraction=FRACTION)

    global BEST_MODEL
    BEST_MODEL = YOLO(model.trainer.best)

    evaluate(dataset.location, name="val-color", split="val")
    show_predictions(
        join(dataset.location, "valid", "images"),
        name="show-color-preds",
    )


if __name__ == "__main__":
    main()
