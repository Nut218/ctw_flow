"""Differentiable cubic anti-aliased resizing for the observation model."""

import numpy as np
import torch
from torch import nn


def cubic(value):
    absolute = np.abs(value)
    squared = absolute ** 2
    cubed = absolute ** 3
    return (
        (1.5 * cubed - 2.5 * squared + 1) * (absolute <= 1)
        + (-0.5 * cubed + 2.5 * squared - 4 * absolute + 2)
        * ((1 < absolute) & (absolute <= 2))
    )


class Resizer(nn.Module):
    def __init__(self, input_shape, scale_factor):
        super().__init__()
        scale_factor = list(scale_factor)
        output_shape = np.uint(np.ceil(np.array(input_shape) * scale_factor))
        self.sorted_dims = [
            int(dim)
            for dim in np.argsort(np.array(scale_factor))
            if scale_factor[dim] != 1
        ]
        fields = []
        weights = []
        for dim in self.sorted_dims:
            weight, field = self._contributions(
                input_shape[dim], output_shape[dim], scale_factor[dim]
            )
            weight = torch.tensor(weight.T, dtype=torch.float32)
            weights.append(
                nn.Parameter(
                    weight.reshape(list(weight.shape) + (len(scale_factor) - 1) * [1]),
                    requires_grad=False,
                )
            )
            fields.append(
                nn.Parameter(
                    torch.tensor(field.T.astype(np.int32), dtype=torch.long),
                    requires_grad=False,
                )
            )
        self.field_of_view = nn.ParameterList(fields)
        self.weights = nn.ParameterList(weights)

    @staticmethod
    def _contributions(input_length, output_length, scale):
        kernel_width = 4.0 / scale
        fixed_kernel = lambda argument: scale * cubic(scale * argument)
        output_coordinates = np.arange(1, output_length + 1)
        shifted = output_coordinates - (output_length - input_length * scale) / 2
        matched = shifted / scale + 0.5 * (1 - 1 / scale)
        left = np.floor(matched - kernel_width / 2)
        expanded_width = np.ceil(kernel_width) + 2
        field = np.squeeze(
            np.int16(np.expand_dims(left, 1) + np.arange(expanded_width) - 1)
        )
        weight = fixed_kernel(np.expand_dims(matched, 1) - field - 1)
        weight = weight / np.expand_dims(
            np.where(weight.sum(1) == 0, 1.0, weight.sum(1)), 1
        )
        mirror = np.uint(
            np.concatenate((np.arange(input_length), np.arange(input_length - 1, -1, -1)))
        )
        field = mirror[np.mod(field, mirror.shape[0])]
        nonzero = np.nonzero(np.any(weight, axis=0))
        return np.squeeze(weight[:, nonzero]), np.squeeze(field[:, nonzero])

    def forward(self, tensor):
        value = tensor
        for dimension, field, weight in zip(
            self.sorted_dims, self.field_of_view, self.weights
        ):
            value = torch.transpose(value, dimension, 0)
            value = torch.sum(value[field] * weight, dim=0)
            value = torch.transpose(value, dimension, 0)
        return value
