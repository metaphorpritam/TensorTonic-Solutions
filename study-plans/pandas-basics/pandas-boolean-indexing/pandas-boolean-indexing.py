import pandas as pd

def boolean_filter(data, column, threshold):
    """
    Returns: dict with 'filtered_data' (dict) and 'count' (int)
    """
    df = pd.DataFrame(data)
    df = df[df[column] > threshold]
    filtered_data = df.to_dict('list')
    count = len(filtered_data[column])

    return {
        "filtered_data": filtered_data,
        "count": count
    }