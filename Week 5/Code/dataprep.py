"""
DESCRIPTION: template script for data preparation.
AUTHORS: JMGG
DATE: 07/10/24
"""

# MODULES IMPORT
from encoder import OneHotEncoder
from textprepar import TextPreparator
from textpreproc import TextPreprocessor
from feeder import CustomDataset
from pandas import read_csv
from os.path import join
from torch.utils.data import DataLoader
from numpy import arange
import torch as th


# DATA LOADING
data_dir = '../Data'
filename = 'obs_lifethread.csv'
path2load = join(data_dir, filename)

data = read_csv(path2load, delimiter=';') #TODO B1.1

# ONE-HOT ENCODING
cols2encode = ['LIFE THREATENING']
encoder = OneHotEncoder()
data = encoder.one_hot_encode(data, cols2encode) #TODO B1.2

# CLASS PREVALENCE
labels_ohe = [col for col in data.columns if 'LAB_' in col]
data_lab = data[labels_ohe]
prevalences = data_lab.sum() / data_lab.shape[0]
print(prevalences)

# TEXT PREPARATION
text_prep = TextPreparator()
data = text_prep.prepare(data['OBSERVATIONS']) #TODO B1.3

# DATA SPLITTING
n_total = len(data)
indexes = arange(n_total)
split_point = int(0.8 * n_total)

train_indexes = indexes[:split_point] #TODO B1.4
test_indexes = indexes[split_point:] #TODO B1.4

data_train = data.iloc[train_indexes]
data_eval = data.iloc[test_indexes]

#TODO B1.5
# TEXT PREPROCESSING
preprocessor = TextPreprocessor()
data_train, indexes_matrix_train = preprocessor.preprocess(
    data=data_train, text_column_identifier='OBSERVATIONS_POSTOK', data_group='training'
)
data_eval, indexes_matrix_eval = preprocessor.preprocess(
    data=data_eval, text_column_identifier='OBSERVATIONS_POSTOK', data_group='evaluation')

# LABELS EXTRACTION
labels_train = data_train[['LAB__NO', 'LAB__YES']].values
labels_eval = data_eval[['LAB__NO', 'LAB__YES']].values

# METADATA EXTRACTION
vocab_size = len(preprocessor.word2index_map.keys())

# TENSOR CONVERSION
indexes_tensor_train = th.from_numpy(indexes_matrix_train).long()
labels_tensor_train = th.from_numpy(labels_train).float()

indexes_tensor_eval = th.from_numpy(indexes_matrix_eval).long()
labels_tensor_eval = th.from_numpy(labels_eval).float()

# DATASETS GENERATION
dataset_train = CustomDataset(features=indexes_tensor_train, labels=labels_tensor_train)
dataset_eval = CustomDataset(features=indexes_tensor_eval, labels=labels_tensor_eval)

# DATA LOADER GENERATION
batch_size_train = 8
batch_size_eval = 64
dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size_train, shuffle=True, drop_last=False)
dataloader_eval = DataLoader(dataset=dataset_eval, batch_size=batch_size_eval, shuffle=False, drop_last=False)
