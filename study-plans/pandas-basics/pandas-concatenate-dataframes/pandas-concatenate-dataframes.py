import pandas as pd

def concat_dataframes(dfs):
    """
    Returns: list [shape, data] where shape is [rows, cols]
    """
    new = pd.DataFrame()
    for df in dfs:
        temp = pd.DataFrame(df)
        new = pd.concat([new, temp], axis = 0, ignore_index=True)

    shape = list(new.shape)
    data = new.to_dict('list')

    return [shape, data]
    
        
        