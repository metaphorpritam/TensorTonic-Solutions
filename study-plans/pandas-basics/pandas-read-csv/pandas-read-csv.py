import pandas as pd

def create_dataframe(data):
    """
    Returns: dict with 'data', 'shape', 'columns'
    """
    df  = pd.DataFrame(data)

    return {
        "data": {col : df[col].to_list() for col in df.columns},
        "shape": list(df.shape),
        "columns": df.columns.to_list()
    }