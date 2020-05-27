##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##########################################################################
# Efficient Neural Architecture Search via Parameters Sharing, ICML 2018 #
##########################################################################
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Text, Dict
from .search_cells import NASNetSearchCell as SearchCell
from .search_model_enas_nasnet_utils import Controller


class NASNetworkENAS(nn.Module):

  def __init__(self, C: int, N: int, steps: int, multiplier: int, stem_multiplier: int,
               num_classes: int, search_space: List[Text], affine: bool, track_running_stats: bool):
    super(NASNetworkENAS, self).__init__()
    self._C        = C
    self._layerN   = N
    self._steps    = steps
    self._multiplier = multiplier
    self.stem = nn.Sequential(
                    nn.Conv2d(3, C*stem_multiplier, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(C*stem_multiplier))
  
    # config for each layer
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * (N-1) + [C*4 ] + [C*4  ] * (N-1)
    layer_reductions = [False] * N + [True] + [False] * (N-1) + [True] + [False] * (N-1)

    num_edge, edge2index = None, None
    C_prev_prev, C_prev, C_curr, reduction_prev = C*stem_multiplier, C*stem_multiplier, C, False

    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      cell = SearchCell(search_space, steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, affine, track_running_stats)
      if num_edge is None: num_edge, edge2index = cell.num_edges, cell.edge2index
      else: assert num_edge == cell.num_edges and edge2index == cell.edge2index, 'invalid {:} vs. {:}.'.format(num_edge, cell.num_edges)
      self.cells.append( cell )
      C_prev_prev, C_prev, reduction_prev = C_prev, multiplier*C_curr, reduction
    self.op_names   = deepcopy( search_space )
    self._Layer     = len(self.cells)
    self.edge2index = edge2index
    self.lastact    = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)
    # to maintain the sampled architecture
    self.sampled_arch = None

  def update_arch(self, _arch):
    if _arch is None:
      self.sampled_arch = None
    elif isinstance(_arch, (list, tuple)):
      self.sampled_arch = _arch
    else:
      raise ValueError('invalid type of input architecture : {:}'.format(_arch))
    return self.sampled_arch
    
  def create_controller(self):
    return Controller(self._steps, len(self.op_names))

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, Max-Nodes={max_nodes}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def forward(self, inputs):
    s0 = s1 = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      if cell.reduction: arc = self.sampled_arch[0]
      else             : arc = self.sampled_arch[1]
      s0, s1 = s1, cell.forward_sample(s0, s1, arc)
    out = self.lastact(s1)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits