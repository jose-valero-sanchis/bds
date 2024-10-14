"""
DESCRIPTION: classes and operations for text preparation.
AUTHORS: ...
DATE: 11/10/21
"""

# MODULES IMPORT
import string
from abc import ABC, abstractmethod
import re
import pandas as pd


# TEXT PREPARATOR
class TextPreparator:

    # TEXT PREPARATION
    # Global text preparation
    @staticmethod
    def prepare(data: pd.DataFrame, text_column_identifier: str) -> pd.DataFrame:
        # Text series extraction
        text_series = data[text_column_identifier]

        # Preparation
        # pretokenization operations
        text_series_pretok = PretokenizationTextPreparator.prepare(text_series=text_series)
        # tokenization
        text_series_tok = TokenizationTextPreparator.prepare(text_series=text_series_pretok)
        # postokenization operations
        text_series_postok = PostokenizationTextPreparator.prepare(text_series=text_series_tok)

        # Arrangement
        data[text_column_identifier + '_PRETOK'] = text_series_pretok
        data[text_column_identifier + '_TOK'] = text_series_tok
        data[text_column_identifier + '_POSTOK'] = text_series_postok

        # Output
        return data

# SUBTASK TEXT PREPARATOR
class SubtaskTextPreparator(ABC):

    # Specific text preparation
    @classmethod
    @abstractmethod
    def prepare(cls, text_series: pd.Series) -> pd.Series:
        raise NotImplementedError

# PRETOKENIZATION TEXT PREPARATION
class PretokenizationTextPreparator(SubtaskTextPreparator):
    # CLASS ATTRIBUTES
    accent_marks_map = {'á': 'a', 'à': 'a', 'é': 'e', 'è': 'e', 'í': 'i', 'ì': 'i', 'ó': 'o', 'ò': 'o',
                        'ú': 'u', 'ù': 'u'}
    punctuation_marks = set(string.punctuation + '\\')
    glued_string_map = {'º': ' º'}

    # TEXT PREPARATION
    @classmethod
    def prepare(cls, text_series: pd.Series) -> pd.Series:
        # Preparation
        # lower case
        text_series_prepared = text_series.apply(cls._lower_case)
        # accent marks
        text_series_prepared = text_series_prepared.apply(cls._process_accent_marks)
        # punctuation marks
        text_series_prepared = text_series_prepared.apply(cls._process_punctuation_marks)
        # glued strings
        text_series_prepared = text_series_prepared.apply(cls._process_glued_strings)
        # trailing newline
        text_series_prepared = text_series_prepared.apply(cls._remove_trailing_newline)

        # Output
        return text_series_prepared

    # LOWER CASE SENTENCE
    @staticmethod
    def _lower_case(line_string: str) -> str:
        line_string_lower = line_string.lower()
        #raise NotImplementedError  # TODO Implement this method
        return line_string_lower

    # ACCENT MARKS
    @classmethod
    def _process_accent_marks(cls, line_string: str) -> str:
        # Initialization
        line_string_processed = ''

        # Processing
        for char in line_string:
            if char not in cls.accent_marks_map.keys():
                line_string_processed += char
            else:
                line_string_processed += cls.accent_marks_map[char]

        # Output
        return line_string_processed

    # PUNCTUATION MARKS
    @classmethod
    def _process_punctuation_marks(cls, line_string: str) -> str:
        #raise NotImplementedError  # TODO Implement this method
        # Extraction of unique sentence characters
        line_string_set = set(line_string)
        # Extraction of punctuation marks present in the sentence
        punc_mark = line_string_set.intersection(cls.punctuation_marks)
        # Processing
        # initialization
        line_string_processed = line_string
        # processing
        for pm in punc_mark:
            line_string_processed = line_string_processed.replace(pm, ' ')
        # Output
        return line_string_processed


    # GLUED STRING
    @classmethod
    def _process_glued_strings(cls, line_string: str) -> str:
        # Initialization
        line_string_processed = ''

        # Processing
        for char in line_string:
            if char not in cls.glued_string_map.keys():
                line_string_processed += char
            else:
                line_string_processed += cls.glued_string_map[char]

        # Output
        return line_string_processed

    # TRAILING NEWLINE
    @staticmethod
    def _remove_trailing_newline(line_string: str) -> str:
        # Processing
        line_processed = line_string.replace('\n', ' ')
        line_processed = line_processed.replace('\r', ' ')
        line_processed = line_processed.strip()

        # Output
        return line_processed

# TOKENIZATION TEXT PREPARATION
class TokenizationTextPreparator(SubtaskTextPreparator):

    # TEXT PREPARATION
    @classmethod
    def prepare(cls, text_series: pd.Series) -> pd.Series:
        # Preparation
        text_series_prepared = text_series.apply(cls._tokenize_word)

        # Output
        return text_series_prepared

    # WORD TOKENIZATION
    @staticmethod
    def _tokenize_word(line_string: str) -> list:
        #raise NotImplementedError  # TODO Implement this method
        # Word tokenization
        list_tokens = re.findall(r"[\w']+", line_string)
        # Output
        return list_tokens

# POSTOKENIZATION TEXT PREPARATION
class PostokenizationTextPreparator(SubtaskTextPreparator):
    # CLASS ATTRIBUTES
    _abbreviation_map_path = '../Data/abbrev2word_map.csv'
    _abbreviation_map = None

    # TEXT PREPARATION
    @classmethod
    def prepare(cls, text_series: pd.Series) -> pd.Series:
        # Preparation
        # map abbreviations
        text_series_prepared = cls._map_abbreviations(text_series)

        # Output
        return text_series_prepared

    # ABBREVIATION MAPPING
    # Series level
    @classmethod
    def _map_abbreviations(cls, text_series: pd.Series) -> pd.Series:
        # Abbreviation frame loading
        # TODO Load abbreviations frame
        abb_frame = pd.read_csv(cls._abbreviation_map_path, delimiter=';', encoding='latin-1', engine='python')

        # Casting to dictionary
        # TODO Transform abbreviations frame into a dictionary
        abb_map = dict(zip(abb_frame['ABBREVIATION'], abb_frame['MEANING']))

        # Inclusion as class attribute
        # TODO Upgrade the class attribute _abbreviation_map
        cls._abbreviation_map = abb_map

        # Mapping
        text_series_mapped = text_series.apply(cls._map_abbreviations_instance)

        # Output
        return text_series_mapped

    # Instance level
    @classmethod
    def _map_abbreviations_instance(cls, line_tokens: list) -> list:
        #raise NotImplementedError  # TODO Implement this method
        # Mapping
        line_tokens_mapped = [cls._abbreviation_map[token] if token in cls._abbreviation_map.keys() else token for token
                              in line_tokens]
        # Output
        return line_tokens_mapped