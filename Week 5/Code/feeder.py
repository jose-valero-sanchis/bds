"""
DESCRIPTION: classes and operations for data feeding.
AUTHORS: ...
DATE: 11/10/21
"""

# MODULES IMPORT
from torch import Tensor
from torch.utils.data import Dataset


# CUSTOM DATASET
class CustomDataset(Dataset):

    # INITIALIZATION
    def __init__(self, features: Tensor, labels: Tensor) -> None:
        # Attributes assignation
        self._features = features
        self._labels = labels
        self._number_data = labels.size()[0]

    # EXTERNAL ATTRIBUTE ACCESS AND EDITION CONTROL
    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def number_data(self):
        return self._number_data

    # ITEM EXTRACTION
    def __getitem__(self, idxs) -> dict:
        # Slicing
        features_sliced = self.features[idxs, :]
        labels_sliced = self.labels[idxs, :]

        # Arrangement
        data_batch = {'features': features_sliced, 'labels': labels_sliced}

        # Output
        return data_batch

    # NUMBER OF DATA EXTRACTION
    def __len__(self):
        return self.number_data
