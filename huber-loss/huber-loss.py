import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float)

    err_abs = np.abs(y_pred - y_true)
    mask = err_abs <= delta
    loss_sample = np.where(mask, 0.5 * (err_abs**2), 
        delta*(err_abs - 0.5 * delta))
    loss = np.mean(loss_sample)
    return loss