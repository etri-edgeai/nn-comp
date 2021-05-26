from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockWiseEmbedding(nn.Module):

    def __init__(
        self,
        assignment,
        block_sizes,
        output_dim,
        compensator=None,
        internal_dim=None,
        embedding_initializer=lambda x:nn.init.normal_(x),
        transformer_initializer=lambda x:nn.init.xavier_uniform_(x)):
        super(BlockWiseEmbedding, self).__init__()

        block_assign_ = torch.zeros(len(assignment), requires_grad=False)
        local_idx_ = torch.zeros(len(assignment), requires_grad=False)
        for idx, block_idx, local_idx in assignment:
            block_assign_[idx] = block_idx
            local_idx_[idx] = local_idx
        self.register_buffer("block_assignment", block_assign_)
        self.register_buffer("local_assignment", local_idx_)

        if internal_dim is None:
            internal_dim = output_dim

        self.blocks = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(int(num), int(size)))
            for num, size in block_sizes
        ])
        self.transformers = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(int(size), internal_dim))
            for num, size in block_sizes
        ])
        self.compensator = compensator

        # init weights
        for b in self.blocks:
            embedding_initializer(b.data)
        for t in self.transformers:
            transformer_initializer(t.data)
        
    def forward(self, src):
        bags = []
        for bidx, data in enumerate(src):
            vecs = []
            for widx, idx in enumerate(data):
                idx = int(idx)
                block_idx = int(self.block_assignment[idx])
                local_idx = int(self.local_assignment[idx])
                vec = self.blocks[block_idx][local_idx]
                vecs.append(torch.matmul(vec, self.transformers[block_idx]))
            bags.append(torch.stack(vecs))
        if self.compensator is not None:
            return self.compensator(torch.stack(bags))
        else:
            return torch.stack(bags)
