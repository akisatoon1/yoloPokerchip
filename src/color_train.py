from os.path import join

from ultralytics import YOLO

import env
import utils
from training import (
    show_predictions,
    train_model,
)


def main():
    dataset = utils.download_dataset(
        env.ROBOFLOW_COLOR_PROJECT,
        version_n=3,
    )
    model = train_model(
        dataset, epochs=env.EPOCHS, patience=env.PATIENCE, fraction=env.FRACTION
    )

    best_model = YOLO(model.trainer.best)

    utils.evaluate(
        name="val-color", model=best_model, dataset_dir=dataset.location, split="val"
    )
    show_predictions(
        join(dataset.location, "valid", "images"),
        name="show-color-preds",
        model=best_model,
    )


if __name__ == "__main__":
    main()
