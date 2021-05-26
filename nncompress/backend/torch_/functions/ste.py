from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

class ChannelMasking(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, mask):
        ctx.save_for_backward(input)
        input = input.transpose(1, -1)
        input= mask * input
        input = input.transpose(1, -1)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input, None
