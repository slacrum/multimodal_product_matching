from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
from models.mnn_em import _CNNBranch
from models.addons.tensorflow_addons.losses import triplet_multimodal


class MNNBTL(object):
    def __init__(self, head_config, char_cnn, learning_rate, name="MNN_BTL"):
        self.head_config = head_config
        self.char_cnn = char_cnn
        self.learning_rate = learning_rate
        self.name = name
        self._build_model()  # builds self.model variable

    def _build_model(self):
        # Text Input
        img_features = Input(shape=(self.head_config["img_input_size"]),
                             name="Image_Input_Head_Outer")

        img_cnn = _CNNBranch(self.head_config["img_input_size"],
                             self.head_config["img_fc_layers"], self.head_config["extended"], True, name="Image")

        # Image Input
        text_features = Input(shape=(self.head_config["txt_input_size"]),
                              name="Text_Input_Head_Outer")

        x = self.char_cnn(text_features)

        text_branch = _CNNBranch(x.shape[1],
                                 self.head_config["txt_fc_layers"], self.head_config["extended"], True, name="Text")

        output_text_branch = text_branch.model(x)

        text_cnn = Model(inputs=text_features,
                         outputs=output_text_branch, name="Text_CNN")

        model = Model(inputs=[img_features, text_features],
                      outputs=Concatenate()(
                          [img_cnn.model(img_features), text_cnn(text_features)]),
                      name=self.name)

        optimizer = Adam(learning_rate=self.learning_rate)

        loss = triplet_multimodal.MultimodalTripletHardLossBidirectional(
            distance_metric="angular")

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=loss)

        self.model = model
