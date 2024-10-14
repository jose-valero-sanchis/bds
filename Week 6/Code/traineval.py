"""
DESCRIPTION: script for modeling.
AUTHORS: ...
DATE: 11/10/21
"""

# MODULES IMPORT
import torch as th
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skmet

# TODO Import your data loaders and vocabulary size data here
from textmodel import TextModel
from dataprep import vocab_size
from dataprep import dataloader_train, dataloader_eval#TODO B2.2

# SETTINGS
# TODO 2.3: Assign values to this hyperparameters
_LEARNING_RATE = 0.00001
_EMBEDDING_DIMENSION = 16 #4, 8, 10, 16
_NUMBER_GRU_UNITS = 2 #1 (fatal) 2, 3 no mejora, 4
_NUMBER_GRU_NEURONS = 32 #64 no bien
_MAXIMUM_EPOCHS = 150

# MODEL INITIALIZATION
model = TextModel(vocabulary_size=vocab_size + 1, embedding_dimension=_EMBEDDING_DIMENSION, padding_index=0,
                  number_GRU_units=_NUMBER_GRU_UNITS, number_GRU_neurons=_NUMBER_GRU_NEURONS, number_classes=2)

# OPTIMIZER
optimizer = Adam(model.parameters(), lr=_LEARNING_RATE)


# LOSS FUNCTION
def cross_entropy(label, output_scores):
    # Calculation
    cross_entropy_tensor = -(label.float() * output_scores.log())
    cross_entropy_sum = th.sum(cross_entropy_tensor, dim=1)
    cross_entropy_mean = cross_entropy_sum.mean()

    # Output
    return cross_entropy_mean


# TRAINING
maximum_epochs = _MAXIMUM_EPOCHS

losses_epochs_train = []
losses_epochs_eval = []

for epoch in range(maximum_epochs):

    losses_epoch_train = []
    losses_epoch_eval = []

    model.train()

    for batch, data_batch_train in enumerate(dataloader_train):
        features_batch_train = data_batch_train['features']
        labels_batch_train = data_batch_train['labels']

        predictions_batch_train = model.forward(features_batch_train)

        loss_batch_train = cross_entropy(labels_batch_train, predictions_batch_train)

        optimizer.zero_grad()

        loss_batch_train.backward()

        optimizer.step()

        losses_epoch_train.append(loss_batch_train.item())

    model.eval()

    with th.no_grad():
        for batch, data_batch_eval in enumerate(dataloader_eval):
            features_batch_eval = data_batch_eval['features']
            labels_batch_eval = data_batch_eval['labels']

            predictions_batch_eval = model.forward(features_batch_eval)

            loss_batch_eval = cross_entropy(labels_batch_eval, predictions_batch_eval)

            losses_epoch_eval.append(loss_batch_eval.item())

    loss_epoch_train = sum(losses_epoch_train) / len(losses_epoch_train)
    loss_epoch_eval = sum(losses_epoch_eval) / len(losses_epoch_eval)

    losses_epochs_train.append(loss_epoch_train)
    losses_epochs_eval.append(loss_epoch_eval)

    print('EPOCH ' + str(epoch) + ' TRAIN: ' + str(loss_epoch_train) + ' EVAL: ' + str(loss_epoch_eval))

# PLOTTING
epochs = np.arange(0, maximum_epochs)
plt.plot(epochs, losses_epochs_train, color='royalblue', linewidth=2)
plt.plot(epochs, losses_epochs_eval, color='orange', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Losses across epochs')
plt.legend(('Training', 'Evaluation'))
plt.grid(True)
plt.show()

# PREDICTION
# Tensors extraction
features_eval = dataloader_eval.dataset.features
labels_eval = dataloader_eval.dataset.labels

# Prediction
predictions_eval = model.forward(features_eval)

# Casting
probs_eval = predictions_eval.detach().numpy()

# Saturation
pred_eval = np.argmax(probs_eval, axis=1)
labs_eval = np.argmax(labels_eval.numpy(), axis=1)

# METRICS CALCULATION
# TODO B2.4: Calculate test metrics
auc = #TODO B2.4: using sklearn.metrics as skmet
accuracy = #TODO B2.4: using sklearn.metrics as skmet
f1_score_macro = #TODO B2.4: using sklearn.metrics as skmet
print(auc, accuracy, f1_score_macro)
