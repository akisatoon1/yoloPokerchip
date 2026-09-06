from ultralytics import YOLO
from os.path import join

from training import (
    download_dataset,
    train_model,
    evaluate,
    show_predictions,
)
import env


def main():
    dataset = download_dataset(
        env.ROBOFLOW_API_KEY,
        env.ROBOFLOW_WORKSPACE,
        env.ROBOFLOW_COLOR_PROJECT,
        version_n=3,
    )
    model = train_model(
        dataset, epochs=env.EPOCHS, patience=env.PATIENCE, fraction=env.FRACTION
    )

    best_model = YOLO(model.trainer.best)

    evaluate(dataset.location, name="val-color", split="val", model=best_model)
    show_predictions(
        join(dataset.location, "valid", "images"),
        name="show-color-preds",
        model=best_model,
    )


if __name__ == "__main__":
    main()
