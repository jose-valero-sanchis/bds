"""
DESCRIPTION: template script for data preparation.
AUTHORS: JMGG
DATE: 07/10/24
"""

# MODULES IMPORT
from torch import Tensor
from torch.nn import Module, GRU, Embedding, Sequential, Linear, BatchNorm1d, ReLU, LeakyReLU, Softmax, Dropout

# TEXT MODEL
class TextModel(Module):

    # INITIALIZATION
    def __init__(self, vocabulary_size: int, embedding_dimension: int, padding_index: int, number_GRU_units: int,
                 number_GRU_neurons: int, number_classes: int):
        # Parent constructor call
        super().__init__()

        # Attributes assignation
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.padding_index = padding_index
        self.number_GRU_units = number_GRU_units
        self.number_GRU_neurons = number_GRU_neurons
        self.number_classes = number_classes

        # Embedding module
        self.embedding = Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dimension,
                                   padding_idx=padding_index)

        # Recurrent module
        self.recurrent = GRU(input_size=embedding_dimension, hidden_size=number_GRU_neurons,
                             num_layers=number_GRU_units, bias=True, batch_first=True, dropout=0.25,
                             bidirectional=False)

        # Output module
        self.output = self._build_output_block()

    # OUTPUT BLOCK BUILDING
    def _build_output_block(self) -> Module:
        # Initialization
        output_module = Sequential()

        # First output block
        # linear layer
        linear_1 = Linear(self.number_GRU_neurons, 16)
        output_module.add_module('linear_1', linear_1)
        # normalizer
        normalizer = BatchNorm1d(num_features=16)
        output_module.add_module('normalizer', normalizer)
        # activation function
        activation_1 = ReLU()  # TODO B2.1 Change this activation function
        output_module.add_module('activation_1', activation_1)
        # dropout
        dropout = Dropout(0.1)
        output_module.add_module('dropout', dropout)

        # Second output block
        # linear layer
        linear_2 = Linear(16, self.number_classes)
        output_module.add_module('linear_2', linear_2)
        # activation function
        activation_2 = Softmax(dim=1)
        output_module.add_module('activation_2', activation_2)

        # Output
        return output_module

    # FORWARD
    def forward(self, indexes_tensor: Tensor) -> Tensor:
        # Embedding module
        embeddings = self.embedding(indexes_tensor)

        # Recurrent module
        out_recurrent_multi, _ = self.recurrent(embeddings)
        out_recurrent = out_recurrent_multi[:, -1, :]

        # Output module
        yhat = self.output(out_recurrent)

        # Output
        return yhat
