import numpy as np

def generate_random_array(shape, kind, seed):
    """
    Returns: 2D ndarray of float64 random values
    """
    rng = np.random.default_rng(seed =  seed)

    match kind:
        case 'uniform':
            return rng.uniform(low = 0, high = 1, size = shape)
        case 'normal':
            return rng.normal(loc = 0, scale = 1, size = shape)
        case _:
            print("{kind} is not supported!")
 