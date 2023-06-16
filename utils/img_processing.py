from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.vgg19 import VGG19
import pandas as pd


def load_img_model(name):
    model_dict = {
        "MobilenetV3large": MobileNetV3Large,
        "MobilenetV3small": MobileNetV3Small,
        "resnet152": ResNet152,
        "VGG19": VGG19
    }
    if name in model_dict:
        img_model = model_dict[name](
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=None, pooling="max", classes=None
        )
        return img_model
    else:
        raise ValueError("Invalid model name: {}".format(name))


def create_embeddings_from(img_model, img, path, batch_size=2048):
    img_all = image_dataset_from_directory(
        path,
        labels=None,
        label_mode=None,
        class_names=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(256, 256),
        shuffle=False,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='nearest',
        follow_links=False,
        crop_to_aspect_ratio=False
    )

    paths = img_all.file_paths          # save paths right before calling predict(), because that information will be lost

    img_all = img_model.predict(img_all)

    idx = get_img_idx(paths, img)

    return img_all[idx]


def get_img_idx(img_all_paths, img_subset):
    # since the image directory may contain more images than we actually use (e.g. due to not using augmentation),
    # we split into `img_all` and `img_subset`,
    # then we select only those img embeddings in `img_all` which are in `img_subset`.
    img_all = pd.Series(img_all_paths, name="path").str.split(
        "/").str[-2:].str.join("/")
    img_all = img_all.to_frame()
    img_all = img_all.reset_index()

    img_subset = img_subset.merge(img_all, how="left", on="path")

    return img_subset["index"]