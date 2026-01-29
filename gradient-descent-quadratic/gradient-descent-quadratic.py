def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    x = x0
    f1_x = 2*a*x + b
    for i in range(steps):
        x = x - lr * f1_x
        f1_x = 2*a*x + b

    return x
    