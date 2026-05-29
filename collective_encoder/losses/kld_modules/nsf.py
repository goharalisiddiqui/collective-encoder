'''
Rational quadratic spline flow implementation adapted from nflows (https://github.com/bayesiains/nflows/blob/v0.14/nflows/transforms/splines/rational_quadratic.py)
Originally based on https://arxiv.org/abs/1906.04032
'''

import torch

def searchsorted(
    bin_locations, 
    inputs, 
    eps=1e-6
):
    bin_locations[..., -1] += eps
    
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1

def nsf_forward(inputs, widths, heights, derivatives, cumwidths, cumheights):
    bin_idx = searchsorted(cumwidths, inputs)[..., None]
    
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]
    
    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)

    numerator = input_heights * (
        input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
    )
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        * theta_one_minus_theta
    )
    outputs = input_cumheights + numerator / denominator

    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, logabsdet

def nsf_inverse(inputs, widths, heights, derivatives, cumwidths, cumheights):
    bin_idx = searchsorted(cumheights, inputs)[..., None]
    
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]
    
    a = (inputs - input_cumheights) * (
        input_derivatives + input_derivatives_plus_one - 2 * input_delta
    ) + input_heights * (input_delta - input_derivatives)
    b = input_heights * input_derivatives - (inputs - input_cumheights) * (
        input_derivatives + input_derivatives_plus_one - 2 * input_delta
    )
    c = -input_delta * (inputs - input_cumheights)

    discriminant = b.pow(2) - 4 * a * c
    assert (discriminant >= 0).all()

    root = (2 * c) / (-b - torch.sqrt(discriminant))
    # root = (- b + torch.sqrt(discriminant)) / (2 * a)
    outputs = root * input_bin_widths + input_cumwidths

    theta_one_minus_theta = root * (1 - root)
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        * theta_one_minus_theta
    )
    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * root.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - root).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, -logabsdet
    