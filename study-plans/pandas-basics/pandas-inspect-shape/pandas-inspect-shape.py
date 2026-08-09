import pandas as pd

def inspect_dataframe(data):
    """
    Returns: dict with 'rows', 'cols' (ints), 'columns' (list),
    'dtypes' (dict), 'total_values' (int)
    """
    df  = pd.DataFrame(data)
    rows = df.shape[0]
    cols = df.shape[1]
    columns = df.columns.tolist()
    dtypes = {col: str(dtypes) for col, dtypes in df.dtypes.items()}
    total_values = df.size

    return {
        "rows": rows,
        "cols": cols,
        "columns": columns,
        "dtypes": dtypes,
        "total_values": df.size
    }