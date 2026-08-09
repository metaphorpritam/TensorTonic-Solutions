import pandas as pd

def reset_index_demo(data, index_col):
    """
    Returns: list [columns_before_reset, columns_after_reset]
    """
    df = pd.DataFrame(data)
 
    df = df.set_index(index_col)
    cols_bef_reset = df.columns.tolist()
    df = df.reset_index()
    cols_aftr_reset = df.columns.tolist()

    return [cols_bef_reset, cols_aftr_reset]