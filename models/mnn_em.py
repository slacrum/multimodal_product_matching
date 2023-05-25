from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision, BinaryAccuracy, CosineSimilarity
from models.char_cnn_zhang import CharCNNZhang


class MNNEM(object):
    def __init__(self, head_config, combined_fc_layers, learning_rate, metrics=["recall", "precision", "binary_accuracy", "cosine_similarity"], loss='binary_crossentropy') -> None:
        self.head_config = head_config
        self.combined_fc_layers = combined_fc_layers
        self.learning_rate = learning_rate
        self.metrics = metrics
        self.loss = loss
        self._build_model()  # builds self.model variable

    def _build_model(self):
        # # Input Layer
        # img_features = Input(shape=(self.img_input_size), name="Image_Input")

        # # fc + ReLU
        # for i, fl in enumerate(self.img_fc_layers[:-1], 1):
        #     x = Dense(fl, activation='relu',
        #               name=f"Image_FC_{i}")(img_features if i == 1 else x)

        # # fc + L2 Norm
        # x = Dense(
        #     self.img_fc_layers[-1], kernel_regularizer='l2', name="Image_FC_last")(img_features if len(self.img_fc_layers) == 1 else x)

        # output_img = BatchNormalization(name="Image_Batch_Normalization")(x)

        # # Input Layer
        # text_features = Input(shape=(self.txt_input_size), name="Text_Input")

        # x = CharCNNZhang(**self.char_cnn_config)

        # x = x.model(text_features)

        # # fc + ReLU
        # for i, fl in enumerate(self.txt_fc_layers[:-1], 1):
        #     x = Dense(fl, activation='relu',
        #               name=f"Text_FC_{i}")(x)

        # # fc + L2 Norm
        # x = Dense(
        #     self.txt_fc_layers[-1], kernel_regularizer='l2', name="Text_FC_last")(text_features if len(self.txt_fc_layers) == 1 else x)

        # output_text = BatchNormalization(name="Text_Batch_Normalization")(x)

        # # Element-wise product
        # combined = Multiply(
        #     name="Element-wise_Multiplication")([output_img, output_text])

        # Input Layer
        img_features = Input(shape=(self.head_config["img_input_size"]), name="Image_Input")

        # Input Layer
        text_features = Input(shape=(self.head_config["txt_input_size"]), name="Text_Input")

        x = MNNEMHead(**self.head_config)

        x = x.model([img_features, text_features])

        # FC Layers
        for i, comb_fl in enumerate(self.combined_fc_layers, 1):
            x = Dense(comb_fl, activation='relu',
                             name=f"Combined_FC_{i}")(x)

        output = Dense(1, activation='sigmoid', name="Sigmoid")(x)
        model = Model(inputs=[img_features, text_features], outputs=output)

        optimizer = Adam(learning_rate=self.learning_rate)

        model.compile(optimizer=optimizer, loss=self.loss,
                      metrics=self.metrics)
        self.model = model
        # print("MNN-EM model built: ")
        # self.model.summary()

class MNNEMHead(object):
    def __init__(self, img_input_size, txt_input_size, txt_weights, char_cnn_config, img_fc_layers, txt_fc_layers):
        self.img_input_size = img_input_size
        self.txt_input_size = txt_input_size
        self.txt_weights = txt_weights
        self.char_cnn_config = char_cnn_config
        self.img_fc_layers = img_fc_layers
        self.txt_fc_layers = txt_fc_layers
        self._build_model()  # builds self.model variable

    def _build_model(self):
        # Input Layer
        img_features = Input(shape=(self.img_input_size))

        # fc + ReLU
        for i, fl in enumerate(self.img_fc_layers[:-1], 1):
            x = Dense(fl, activation='relu',
                      name=f"Image_FC_{i}")(img_features if i == 1 else x)

        # fc + L2 Norm
        x = Dense(
            self.img_fc_layers[-1], kernel_regularizer='l2', name="Image_FC_last")(img_features if len(self.img_fc_layers) == 1 else x)

        output_img = BatchNormalization(name="Image_Batch_Normalization")(x)

        # Input Layer
        text_features = Input(shape=(self.txt_input_size))

        x = CharCNNZhang(**self.char_cnn_config)

        x = x.model(text_features)

        # fc + ReLU
        for i, fl in enumerate(self.txt_fc_layers[:-1], 1):
            x = Dense(fl, activation='relu',
                      name=f"Text_FC_{i}")(x)

        # fc + L2 Norm
        x = Dense(
            self.txt_fc_layers[-1], kernel_regularizer='l2', name="Text_FC_last")(text_features if len(self.txt_fc_layers) == 1 else x)

        output_text = BatchNormalization(name="Text_Batch_Normalization")(x)

        # Element-wise product
        combined = Multiply(
            name="Element-wise_Multiplication")([output_img, output_text])
        
        model = Model(inputs=[img_features, text_features], outputs=combined, name="MNN_EM_Head")

        self.model = model