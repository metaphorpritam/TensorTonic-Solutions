import pandas as pd

def drop_duplicates(data):
    """
    Returns: list [rows_before, rows_after, cleaned_data]
    """
    df = pd.DataFrame(data)

    rows_before = df.shape[0]
    df = df.drop_duplicates(keep = "first")
    rows_after = df.shape[0]

    cleaned_data = df.to_dict('list')

    return [rows_before, rows_after, cleaned_data]