from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Convolution1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Activation


class CharCNNZhang(object):
    """
    Class to implement the Character Level Convolutional Neural Network for Text Classification,
    as described in Zhang et al., 2015 (http://arxiv.org/abs/1509.01626)
    """

    def __init__(self, input_size, embedding_size,
                 conv_layers, fc_layers, output_size, embedding_weights):
        """
        Initialization for the Character Level CNN model.

        Args:
            input_size (int): Size of input features
            embedding_size (int): Size of embeddings
            conv_layers (list[list[int]]): List of Convolution layers for model
            fc_layers (list[list[int]]): List of Fully Connected layers for model
            output_size (int): Size of output features
        """
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.conv_layers = conv_layers
        self.fc_layers = fc_layers
        self.output_size = output_size
        self.embedding_weights = embedding_weights
        self._build_model()  # builds self.model variable

    def _build_model(self):
        """
        Build and compile the Character Level CNN model

        Returns: None

        """
        # Input layer
        inputs = Input(shape=(self.input_size,),
                       name='sent_input', dtype='int64')
        # Embedding layers
        x = Embedding(self.embedding_size + 1,
                      self.embedding_size,
                      input_length=self.input_size,
                      weights=[self.embedding_weights])(inputs)
        # Convolution layers
        for cl in self.conv_layers:
            x = Convolution1D(cl[0], cl[1])(x)
            x = Activation("relu")(x)
            if cl[2] != -1:
                x = MaxPooling1D(cl[2])(x)
        x = Flatten()(x)
        # Fully connected layers
        for fl in self.fc_layers:
            x = Dense(fl, activation='relu')(x)
        # Output layer
        predictions = Dense(self.output_size, activation='relu')(x)
        # Build and compile model
        model = Model(inputs=inputs, outputs=predictions,
                      name="Character_Level_CNN")
        self.model = model
        # print("CharCNNZhang model built: ")
        # self.model.summary()
