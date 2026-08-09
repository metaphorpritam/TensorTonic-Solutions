import pandas as pd

def create_dataframe(data):
    """
    Returns: dict with 'data', 'shape', 'columns'
    """
    df = pd.DataFrame(data)

    shape = list(df.shape)
    columns = df.columns.to_list()
    data_out = df.to_dict('list')

    return {
        "data": data_out,
        "shape": shape,
        "columns": columns
    }