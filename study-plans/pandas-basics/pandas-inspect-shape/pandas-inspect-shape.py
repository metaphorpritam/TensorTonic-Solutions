import pandas as pd

def inspect_dataframe(data):
    """
    Returns: dict with 'rows', 'cols' (ints), 'columns' (list),
    'dtypes' (dict), 'total_values' (int)
    """
    df = pd.DataFrame(data)
    rows, cols = df.shape
    return {
        "rows": rows,
        "cols": cols,
        "columns": df.columns.to_list(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "total_values": rows * cols
    }