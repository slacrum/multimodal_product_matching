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
    def __init__(self, img_input_size, txt_input_size, img_fc_layers, txt_fc_layers, extended, char_cnn):
        self.img_input_size = img_input_size
        self.txt_input_size = txt_input_size
        self.img_fc_layers = img_fc_layers
        self.txt_fc_layers = txt_fc_layers
        self.extended = extended
        self.char_cnn = char_cnn
        self._build_model()  # builds self.model variable

    def _build_model(self):
        # Text Input
        img_features = Input(shape=(self.img_input_size), name="Image_Input_Head_Outer")

        img_cnn = CNNBranch(self.img_input_size, self.img_fc_layers, self.extended, name="Image")

        output_img = img_cnn.model(img_features)


        # Image Input
        text_features = Input(shape=(self.txt_input_size), name="Text_Input_Head_Outer")
    
        x = self.char_cnn(text_features)

        text_branch = CNNBranch(x.shape[1], self.txt_fc_layers, self.extended, name="Text")

        output_text_branch = text_branch.model(x)

        text_cnn = Model(inputs=text_features, outputs=output_text_branch, name="Text_CNN")

        output_text = text_cnn(text_features)

        # Element-wise product
        combined = Multiply(
            name="Element-wise_Multiplication")([output_img, output_text])
        
        model = Model(inputs=[img_features, text_features], outputs=combined, name="MNN_EM_Head")

        self.model = model

class CNNBranch(object):
    def __init__(self, input_size, fc_layers, extended, name):
        self.input_size = input_size
        self.fc_layers = fc_layers
        self.extended = extended
        self.name = name
        self._build_model()  # builds self.model variable

    def _build_model(self):
        # Input Layer
        features = Input(shape=(self.input_size), name=f"{self.name}_Input_Head_Inner")

        # fc + ReLU
        for i, fl in enumerate(self.fc_layers[:None if self.extended else -1], 1):
            x = Dense(fl, activation='relu',
                      name=f"{self.name}_FC_{i}")(features if i == 1 else x)
            
        if not self.extended:
            # fc + L2 Norm
            x = Dense(
                self.fc_layers[-1], kernel_regularizer='l2', name=f"{self.name}_FC_last")(features if len(self.fc_layers) == 1 else x)

        output = BatchNormalization(name=f"{self.name}_Batch_Normalization")(x)

        model = Model(inputs=features, outputs=output, name=f"{self.name}_CNN")

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