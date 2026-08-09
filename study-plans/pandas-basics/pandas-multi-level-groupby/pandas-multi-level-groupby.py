import pandas as pd

def multi_groupby(data, group_cols, value_col, aggfunc):
    """
    Returns: dict of lists (flat table with group columns + value column)
    """
    df = pd.DataFrame(data)

    out = df.groupby(group_cols)[value_col].agg(aggfunc).reset_index().to_dict('list')

    #out = pd.crosstab(
        #df[value_col],
        #df[group_cols],
        #aggfunc = aggfunc
    #).to_dict()

    return out 

    