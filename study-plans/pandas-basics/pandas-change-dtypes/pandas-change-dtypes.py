import pandas as pd

def change_dtype(data, column, target_type):
    """
    Returns: list [dtypes_before, dtypes_after] (both dicts)
    """
    df = pd.DataFrame(data)
    dtypes_before = {col:str(dtype) for col, dtype in df.dtypes.items()}
    df = df.astype({column: target_type})
    dtypes_after = {col:str(dtype) for col, dtype in df.dtypes.items()}

    return [dtypes_before, dtypes_after]
    