def denormalize_parameters(weights, bias, X_mean, X_std, y_mean, y_std):
    """
    Convert normalized model parameters back to physical scale.
    
    weights: tensor of shape [1, 2] → [w_t, w_t2]
    bias: scalar tensor
    X_mean, X_std: tensors of shape [2] → mean and std of [t, t^2]
    y_mean, y_std: scalars → mean and std of y
    """
    w_t, w_t2 = weights[0]
    mu_t, mu_t2 = X_mean
    sigma_t, sigma_t2 = X_std

    # Recover physical coefficients
    b_phys = (w_t * y_std) / sigma_t
    a_phys = -(w_t2 * y_std) / sigma_t2  # negative because y = -a t^2 + b t + c

    # Recover bias term
    c_phys = y_std * (bias - (w_t * mu_t / sigma_t) - (w_t2 * mu_t2 / sigma_t2)) + y_mean

    return a_phys.item(), b_phys.item(), c_phys.item()
