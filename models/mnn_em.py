from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision, BinaryAccuracy, CosineSimilarity
from models.char_cnn_zhang import CharCNNZhang


class MNNEM(object):
    def __init__(self, img_input_size, img_conv_layers, txt_input_size, txt_conv_layers, txt_weights, char_cnn_config, combined_conv_layers, learning_rate, loss='binary_crossentropy') -> None:
        self.img_input_size = img_input_size
        self.img_conv_layers = img_conv_layers
        self.txt_input_size = txt_input_size
        self.txt_conv_layers = txt_conv_layers
        self.txt_weights = txt_weights
        self.char_cnn_config = char_cnn_config
        self.combined_conv_layers = combined_conv_layers
        self.learning_rate = learning_rate
        self.loss = loss
        self._build_model()  # builds self.model variable

    def _build_model(self):
        # Input Layer
        img_features = Input(shape=(self.img_input_size), name="Image_Input")

        # fc + ReLU
        for i, img_cl in enumerate(self.img_conv_layers[:-1], 1):
            x = Dense(img_cl, activation='relu',
                      name=f"Image_FC_{i}")(img_features if i == 1 else x)

        # fc + L2 Norm
        x = Dense(
            self.img_conv_layers[-1], kernel_regularizer='l2', name="Image_FC_last")(x)

        output_img = BatchNormalization(name="Image_Batch_Normalization")(x)

        # Input Layer
        text_features = Input(shape=(self.txt_input_size), name="Text_Input")

        x = CharCNNZhang(**self.char_cnn_config)

        x = x.model(text_features)

        # fc + ReLU
        for i, txt_cl in enumerate(self.txt_conv_layers[:-1], 1):
            x = Dense(txt_cl, activation='relu',
                      name=f"Text_FC_{i}")(x)

        # fc + L2 Norm
        x = Dense(
            self.txt_conv_layers[-1], kernel_regularizer='l2', name="Text_FC_last")(x)

        output_text = BatchNormalization(name="Text_Batch_Normalization")(x)

        # Element-wise product
        combined = Multiply(
            name="Element-wise_Multiplication")([output_img, output_text])
        # FC Layers
        for i, comb_cl in enumerate(self.combined_conv_layers, 1):
            combined = Dense(comb_cl, activation='relu',
                             name=f"Combined_FC_{i}")(combined)

        output = Dense(1, activation='sigmoid', name="Sigmoid")(combined)
        model = Model(inputs=[img_features, text_features], outputs=output)

        optimizer = Adam(learning_rate=self.learning_rate)

        metrics = [
        Recall(thresholds=0.5, top_k=1, class_id=None, name="recall", dtype=None),
        Precision(thresholds=0.5, top_k=1, class_id=None, name="precision", dtype=None),
        BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5),
        CosineSimilarity(name='cosine_similarity', dtype=None, axis=-1)
        ]

        model.compile(optimizer=optimizer, loss=self.loss,
                      metrics=metrics)
        self.model = model
        # print("MNN-EM model built: ")
        # self.model.summary()
