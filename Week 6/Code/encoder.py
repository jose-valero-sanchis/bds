"""
DESCRIPTION: classes and operations for one-hot encoding
AUTHORS: ...
DATE: 11/10/21
"""

# MODULES IMPORT
import pandas as pd


# ONE HOT ENCODING
class OneHotEncoder:

    @staticmethod
    def one_hot_encode(data: pd.DataFrame, columns2encode: list) -> pd.DataFrame:
        # Data selection
        data2encode = data[columns2encode]

        # Data encoding
        data_encoded = pd.get_dummies(data2encode, prefix='LAB_', drop_first=False)

        # Deletion of original columns
        data = data.drop(columns=columns2encode)

        # Inclusion of generated columns in the original data
        data = pd.concat([data, data_encoded], axis=1)

        # Output
        return data
