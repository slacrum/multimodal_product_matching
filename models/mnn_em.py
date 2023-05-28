from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision, BinaryAccuracy, CosineSimilarity
from models.char_cnn_zhang import CharCNNZhang


class MNNEM(object):
    def __init__(self, head_config, char_cnn, combined_fc_layers, learning_rate, metrics=["recall", "precision", "binary_accuracy", "cosine_similarity"], loss='binary_crossentropy', name="MNN_EM") -> None:
        self.head_config = head_config
        self.char_cnn = char_cnn
        self.combined_fc_layers = combined_fc_layers
        self.learning_rate = learning_rate
        self.metrics = metrics
        self.loss = loss
        self.name = name
        self._build_model()  # builds self.model variable

    def _build_model(self):
        # Input Layer
        img_features = Input(shape=(self.head_config["img_input_size"]), name="Image_Input")

        # Input Layer
        text_features = Input(shape=(self.head_config["txt_input_size"]), name="Text_Input")

        x = MNNEMHead(**self.head_config, char_cnn=self.char_cnn)

        x = x.model([img_features, text_features])

        # FC Layers
        for i, comb_fl in enumerate(self.combined_fc_layers, 1):
            x = Dense(comb_fl, activation='relu',
                             name=f"Combined_FC_{i}")(x)

        output = Dense(1, activation='sigmoid', name="Sigmoid")(x)
        model = Model(inputs=[img_features, text_features], outputs=output, name=self.name)

        optimizer = Adam(learning_rate=self.learning_rate)

        model.compile(optimizer=optimizer, loss=self.loss,
                      metrics=self.metrics)
        self.model = model
        # print("MNN-EM model built: ")
        # self.model.summary()

class MNNEMHead(object):
    def __init__(self, img_input_size, txt_input_size, txt_weights, img_fc_layers, txt_fc_layers, extended, char_cnn):
        self.img_input_size = img_input_size
        self.txt_input_size = txt_input_size
        self.txt_weights = txt_weights
        self.img_fc_layers = img_fc_layers
        self.txt_fc_layers = txt_fc_layers
        self.extended = extended
        self.char_cnn = char_cnn
        self._build_model()  # builds self.model variable

    def _build_model(self):
        # Input Layer
        img_features = Input(shape=(self.img_input_size))

        # fc + ReLU
        for i, fl in enumerate(self.img_fc_layers[:None if self.extended else -1], 1):
            x = Dense(fl, activation='relu',
                      name=f"Image_FC_{i}")(img_features if i == 1 else x)
            
        if not self.extended:
            # fc + L2 Norm
            x = Dense(
                self.img_fc_layers[-1], kernel_regularizer='l2', name="Image_FC_last")(img_features if len(self.img_fc_layers) == 1 else x)

        output_img = BatchNormalization(name="Image_Batch_Normalization")(x)


        # Input Layer
        text_features = Input(shape=(self.txt_input_size))
    
        x = self.char_cnn(text_features)

        # fc + ReLU
        for i, fl in enumerate(self.txt_fc_layers[:None if self.extended else -1], 1):
            x = Dense(fl, activation='relu',
                      name=f"Text_FC_{i}")(x)

        if not self.extended:
            # fc + L2 Norm
            x = Dense(
                self.txt_fc_layers[-1], kernel_regularizer='l2', name="Text_FC_last")(text_features if len(self.txt_fc_layers) == 1 else x)

        output_text = BatchNormalization(name="Text_Batch_Normalization")(x)

        # Element-wise product
        combined = Multiply(
            name="Element-wise_Multiplication")([output_img, output_text])
        
        model = Model(inputs=[img_features, text_features], outputs=combined, name="MNN_EM_Head")

        self.model = model

class ExtendedMNNEM(object):
    def __init__(self, head_1_config, head_2_config, char_cnn, combined_fc_layers, learning_rate, metrics=["recall", "precision", "binary_accuracy", "cosine_similarity"], loss='binary_crossentropy') -> None:
        self.head_1_config = head_1_config
        self.head_2_config = head_2_config
        self.char_cnn = char_cnn
        self.combined_fc_layers = combined_fc_layers
        self.learning_rate = learning_rate
        self.metrics = metrics
        self.loss = loss
        self._build_model()  # builds self.model variable

    def _build_model(self):
        head = MNNEMHead(**self.head_1_config, char_cnn=self.char_cnn)

        tail = MNNEM(
            head_config=self.head_2_config,
            char_cnn=self.char_cnn,
            combined_fc_layers=self.combined_fc_layers,
            learning_rate=self.learning_rate,
            metrics=self.metrics,
            name="MNN_EM_Tail")

        # Input Layer
        img_features = Input(shape=(self.head_1_config["img_input_size"]), name="Image_Input")

        # Input Layer
        text_features = Input(shape=(self.head_1_config["txt_input_size"]), name="Text_Input")

        # Input Layer
        text_2_features = Input(shape=(self.head_2_config["txt_input_size"]), name="Text_2_Input")

        x = head.model([img_features, text_features])

        x = tail.model([x, text_2_features])

        model = Model(inputs=[img_features, text_features, text_2_features], outputs=x, name="Extended_MNN_EM")

        optimizer = Adam(learning_rate=self.learning_rate)

        model.compile(optimizer=optimizer, loss=self.loss,
                      metrics=self.metrics)
        
        self.model = model