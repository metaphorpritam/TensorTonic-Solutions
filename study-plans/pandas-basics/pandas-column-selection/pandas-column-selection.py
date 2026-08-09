import pandas as pd

def select_column(data, column):
    """
    Returns: dict with 'values' (list) and 'length' (int)
    """
    df = pd.DataFrame(data)
    values = df[column].tolist()
    length = len(values)

    return {
        "values": values,
        "length": length
    }