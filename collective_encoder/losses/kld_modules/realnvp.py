
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

from .nsf import nsf_forward, nsf_inverse

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

def confine_to_domain(
    widths: torch.Tensor,
    min_bin_width: float,
    lower_bound: float,
    upper_bound: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_bins = widths.shape[-1]
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths        # Ensure minimum bin width and that widths sum to 1
    cumwidths = torch.cumsum(widths, dim=-1)                                # Cumulative widths for determining bin boundaries
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)    # Pad with zero at the beginning to represent the left boundary
    cumwidths = (upper_bound - lower_bound) * cumwidths + lower_bound       # Scale and shift to fit the specified domain
    cumwidths[..., 0] = lower_bound                                         # Ensure the first cumulative width is exactly the left boundary
    cumwidths[..., -1] = upper_bound                                        # Ensure the last cumulative width is exactly the right boundary                         
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]                       # Recalculate widths to ensure they sum to the correct total after enforcing minimum width 
    
    return widths, cumwidths
    

class RealNVP(nn.Module):
    """
    An implementation of Real NVP.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(
        self, 
        mask, 
        hidden_units=20, 
        spline_knots=31, 
        domain=(-10.0, 10.0),
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE
    ):
        super().__init__()
        num_transformed = int(mask.sum())
        num_identity = mask.shape[0] - num_transformed
        num_bins = spline_knots - 1
        
        if min_bin_width * num_bins > 1.0:
            raise ValueError("Minimal bin width too large for the number of bins")
        if min_bin_height * num_bins > 1.0:
            raise ValueError("Minimal bin height too large for the number of bins")

        self.transform_nn = nn.Sequential(
            nn.Linear(num_identity, hidden_units),
            nn.ReLU(),
        )
        self.widths_layer = nn.Sequential(
            nn.Linear(hidden_units, num_transformed * (spline_knots - 1)),
            nn.Softmax(dim=-1)
        )
        self.heights_layer = nn.Sequential(
            nn.Linear(hidden_units, num_transformed * (spline_knots - 1)),
            nn.Softmax(dim=-1)
        )
        self.derivatives_layer = nn.Sequential(
            nn.Linear(hidden_units, num_transformed * (spline_knots - 1)),
            nn.Softplus()
        )
        
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.mask = mask
        self.domain = domain
    
    def network(self, inputs):
        min_bin_width, min_bin_height, min_derivative, num_bins = (
            self.min_bin_width, 
            self.min_bin_height, 
            self.min_derivative, 
            self.num_bins
        )
        left, right = self.domain
        
        x = inputs * self.mask
        x = self.transform_nn(x)
        widths, heights, derivatives = self.widths_layer(x), self.heights_layer(x), self.derivatives_layer(x)
        
        widths, cumwidths = confine_to_domain(widths, min_bin_width, left, right)
        heights, cumheights = confine_to_domain(heights, min_bin_height, left, right)
        derivatives = min_derivative + derivatives
        
        return widths, heights, derivatives, cumwidths, cumheights
        
    def forward(self, inputs):
        return nsf_forward(inputs, *self.network(inputs))
    
    def inverse(self, inputs):
        return nsf_inverse(inputs, *self.network(inputs))
        
        
        

    # def _log_prob(self, inputs, context):
    #     embedded_context = self._embedding_net(context)
    #     noise, logabsdet = self._transform(inputs, context=embedded_context)
    #     log_prob = self._distribution.log_prob(noise, context=embedded_context)
    #     return log_prob + logabsdet
    
    # def sample(self, num_samples, context=None, batch_size=None):
    #     """Generates samples from the distribution. Samples can be generated in batches.

    #     Args:
    #         num_samples: int, number of samples to generate.
    #         context: Tensor or None, conditioning variables. If None, the context is ignored.
    #         batch_size: int or None, number of samples per batch. If None, all samples are generated
    #             in one batch.

    #     Returns:
    #         A Tensor containing the samples, with shape [num_samples, ...] if context is None, or
    #         [context_size, num_samples, ...] if context is given.
    #     """
    #     if context is not None:
    #         context = torch.as_tensor(context)

    #     if batch_size is None:
    #         return self._sample(num_samples, context)

    #     else:

    #         num_batches = num_samples // batch_size
    #         num_leftover = num_samples % batch_size
    #         samples = [self._sample(batch_size, context) for _ in range(num_batches)]
    #         if num_leftover > 0:
    #             samples.append(self._sample(num_leftover, context))
    #         return torch.cat(samples, dim=0)

    # def _sample(self, num_samples, context):
    #     embedded_context = self._embedding_net(context)
    #     noise = self._distribution.sample(num_samples, context=embedded_context)

    #     if embedded_context is not None:
    #         # Merge the context dimension with sample dimension in order to apply the transform.
    #         noise = torchutils.merge_leading_dims(noise, num_dims=2)
    #         embedded_context = torchutils.repeat_rows(
    #             embedded_context, num_reps=num_samples
    #         )

    #     samples, _ = self._transform.inverse(noise, context=embedded_context)

    #     if embedded_context is not None:
    #         # Split the context dimension from sample dimension.
    #         samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])

    #     return samples

    # def sample_and_log_prob(self, num_samples, context=None):
    #     """Generates samples from the flow, together with their log probabilities.

    #     For flows, this is more efficient that calling `sample` and `log_prob` separately.
    #     """
    #     embedded_context = self._embedding_net(context)
    #     noise, log_prob = self._distribution.sample_and_log_prob(
    #         num_samples, context=embedded_context
    #     )

    #     if embedded_context is not None:
    #         # Merge the context dimension with sample dimension in order to apply the transform.
    #         noise = torchutils.merge_leading_dims(noise, num_dims=2)
    #         embedded_context = torchutils.repeat_rows(
    #             embedded_context, num_reps=num_samples
    #         )

    #     samples, logabsdet = self._transform.inverse(noise, context=embedded_context)

    #     if embedded_context is not None:
    #         # Split the context dimension from sample dimension.
    #         samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
    #         logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

    #     return samples, log_prob - logabsdet

    # def transform_to_noise(self, inputs, context=None):
    #     """Transforms given data into noise. Useful for goodness-of-fit checking.

    #     Args:
    #         inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
    #         context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
    #             with the data.

    #     Returns:
    #         A `Tensor` of shape [batch_size, ...], the noise.
    #     """
    #     noise, _ = self._transform(inputs, context=self._embedding_net(context))
    #     return noise
    
    # def mean(self, context=None):
    #     if context is not None:
    #         context = torch.as_tensor(context)
    #     return self._mean(context)

    # def _mean(self, context):
    #     raise NoMeanException()