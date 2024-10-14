"""
DESCRIPTION: classes and operations for text prepocessing.
AUTHORS: ...
DATE: 11/10/21
"""

# MODULES IMPORT
import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.preprocessing import sequence

# TEXT PREPROCESSOR
class TextPreprocessor:
    # CLASS ATTRIBUTES
    _data_groups = ('training', 'evaluation')
    _minimum_number_occurrences = 3
    _unknown_token = '$unknown$'
    _pad_token = '$pad$'
    _sentence_length = 30
    _padding_type = 'pre'  # 'pre', 'post'
    _truncating_type = 'post'  # 'pre', 'post'

    # INITIALIZATION
    def __init__(self) -> None:

        # Attributes assignation
        self._word_counts = None
        self._word2index_map = None
        self._sentence_lengths = None

    # EXTERNAL ATTRIBUTE ACCESS AND EDITION CONTROL
    # Word counts
    @property
    def word_counts(self) -> dict:
        return self._word_counts

    @word_counts.setter
    def word_counts(self, value: dict) -> None:
        if type(value) is not dict:
            raise ValueError('Word counts must be specified as a dictionary.')

        self._word_counts = value

    # Word to index map
    @property
    def word2index_map(self) -> dict:
        return self._word2index_map

    @word2index_map.setter
    def word2index_map(self, value: dict) -> None:
        if type(value) is not dict:
            raise ValueError('Word to index map must be specified as a dictionary.')

        self._word2index_map = value

    # Sentence lengths
    @property
    def sentence_lengths(self) -> np.array:
        return self._sentence_lengths

    @sentence_lengths.setter
    def sentence_lengths(self, value: np.array) -> None:
        if type(value) is not np.array:
            raise ValueError('Sentence lengths must be specified as a numpy array.')

        self._sentence_lengths = value

    # TEXT PREPROCESSING
    def preprocess(self, data: pd.DataFrame, text_column_identifier: str, data_group: str) -> (pd.DataFrame, np.array):

        # Data group checking
        if data_group not in self._data_groups:
            raise ValueError('Unrecognized data group identifier.')

        # Text series extraction
        text_series = data[text_column_identifier]

        # Word counts extraction
        if data_group == 'training':
            self._get_word_counts(text_series)

        # Word to index map extraction
        if data_group == 'training':
            self._get_word2index_map()

        # Word to index mapping
        data[text_column_identifier + '_IDXS'] = self._word2index_mapping(text_series)

        # Sentence lengths extraction
        if data_group == 'training':
            data[text_column_identifier + '_SLEN'] = self._get_sentence_lengths(data[text_column_identifier + '_IDXS'])

        # Sequence padding and truncation
        indexes_matrix = self._pad_truncate_sequence(data[text_column_identifier + '_IDXS'])

        # Output
        return data, indexes_matrix

    # WORD COUNTS EXTRACTION
    def _get_word_counts(self, text_series: pd.Series) -> None:
        # Word counts dict extraction
        # TODO Complete
        # Texts merging
        text_merged = text_series.sum()

        # Word counts extraction
        # initialization
        word_counts = {}

        # extraction
        for word in text_merged:
            if word not in word_counts:
                word_counts[word] = 1

            else:
                word_counts[word] += 1

        # Word counts attribute upgrading
        self.word_counts = word_counts


    # WORD TO INDEX MAP EXTRACTION
    def _get_word2index_map(self):
        # Extraction from word counts
        # TODO Generate a dictionary (named word2index_map) that maps each word to an index. You have just to consider
        # TODO those words whose absolute frequency is equal or greater than the specified in the class attribute
        # TODO _minimum_number_occurrences. Your indexes must be positive integers, starting by 1.
        # Initialization
        # word2index map
        word2index_map = {}
        # counter
        counter = 1

        # Extraction from word counts
        for word, counts in self.word_counts.items():
            if counts >= self._minimum_number_occurrences:
                word2index_map[word] = counter
                counter += 1

        # Inclusion of padding index
        word2index_map[self._pad_token] = 0

        # Inclusion of unknown index
        vocab_size = len(word2index_map.keys())
        word2index_map[self._unknown_token] = vocab_size + 1

        # Word to index attribute upgrading
        self.word2index_map = word2index_map

    # WORD TO INDEX MAPPING
    # Series level
    def _word2index_mapping(self, text_series: pd.Series) -> pd.Series:
        # Word to index mapping
        text_series_mapped = text_series.apply(self._word2index_mapping_instance)

        # Output
        return text_series_mapped

    # Instance level
    def _word2index_mapping_instance(self, line_tokens: list) -> list:
        #raise NotImplementedError  # TODO Implement this method
        # Initialization
        line_tokens_mapped = []

        # Mapping
        for token in line_tokens:
            if token in self.word2index_map.keys():
                line_tokens_mapped.append(self.word2index_map[token])

            else:
                line_tokens_mapped.append(self.word2index_map[self._unknown_token])

        # Output
        return line_tokens_mapped


    # SENTENCE LENGTHS EXTRACTION
    def _get_sentence_lengths(self, indexes_series: pd.Series) -> pd.Series:
        return indexes_series.apply(lambda x: len(x))

    # PADDING AND TRUNCATION
    def _pad_truncate_sequence(self, indexes_series: pd.Series) -> np.array:
        # Padding and truncation
        indexes_matrix = sequence.pad_sequences(indexes_series.to_list(), maxlen=self._sentence_length,
                                              padding=self._padding_type, truncating=self._truncating_type)

        # Output
        return indexes_matrix
