import pandas as pd

def apply_transform(data, column, operation):
    df = pd.DataFrame(data)
    new_col = str(column) + "_transformed"

    match operation:
        case "normalize":
            df[new_col] = ((df[column] - df[column].min()) /
                            (df[column].max() - df[column].min())).round(4)
        case "double":
            df[new_col] = df[column] * 2
        case "rank":
            df[new_col] = df[column].rank().astype(int)
        case "cumsum":
            df[new_col] = df[column].cumsum()
        case _:
            raise ValueError(f"Unknown operation: {operation}")

    return df.to_dict('list')