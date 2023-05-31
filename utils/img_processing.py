from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.vgg19 import VGG19


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
    datagenerator = ImageDataGenerator()

    img = datagenerator.flow_from_dataframe(
        dataframe=img, directory=path, x_col='path', y_col=None,
        weight_col=None, target_size=(256, 256), color_mode='rgb',
        classes=None, class_mode="input", batch_size=batch_size, shuffle=False,
        seed=None, save_to_dir=None, save_prefix='',
        save_format='jpg', subset=None, interpolation='nearest',
        validate_filenames=True
    )

    return img_model.predict(img)
