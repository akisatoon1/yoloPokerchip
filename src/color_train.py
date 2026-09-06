from os.path import join

from ultralytics import YOLO

import env
import utils


def main():
    dataset = utils.download_dataset(
        env.ROBOFLOW_COLOR_PROJECT,
        version_n=3,
    )
    model = utils.train_model(name="color-trained", dataset=dataset)

    best_model = YOLO(model.trainer.best)

    utils.evaluate(
        name="val-color", model=best_model, dataset_dir=dataset.location, split="val"
    )
    utils.show_predictions(
        name="show-color-preds",
        model=best_model,
        imgs_dir=join(dataset.location, "valid", "images"),
    )


if __name__ == "__main__":
    main()
