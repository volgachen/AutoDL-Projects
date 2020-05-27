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
from log_utils    import AverageMeter, time_string
from utils import obtain_accuracy

import random, time


def TrackRunningStats(module):
  if isinstance(module, nn.BatchNorm2d):
    module.track_running_stats = True


def ResetRunningStats(module):
  if isinstance(module, nn.BatchNorm2d):
    module.reset_running_stats()

def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func


class Searcher(object):
  def __init__(self, model, train_loader, val_loader, logger, config):
    self.logger = logger
    self.max_epochs  = config.search_epoch
    self.population_num = config.population_num
    self.parent_num = config.parent_num
    self.flops_limit = 300 * 1e6

    self.model = model
    self.train_loader, self.val_loader = train_loader, val_loader

    self.perform_dict = {}
    if config.frozen_bn:
        self.train_loader = None
        self.logger.log('Log Info: Searching with BN frozen.')

    #record
    self.best_cand = None
    self.best_perf = None

    # Inference Time Limit
    self.max_inference_time = 0.06
    self.want_inference_time = 0.04
    self.lambda_t = config.lambda_t

  @no_grad_wrapper
  def test_archi_acc(self, arch):
    if self.train_loader is not None:
        self.model.apply(ResetRunningStats)

        self.model.train()
        for step, (data, target) in enumerate(self.train_loader):
            # print('train step: {} total: {}'.format(step,max_train_iters))
            # data, target = train_dataprovider.next()
            # print('get data',data.shape)
            #data = data.cuda()
            output = self.model.forward(data, arch)#_with_architect
            del data, target, output


    base_top1, base_top5 = AverageMeter(), AverageMeter()
    self.model.eval()

    one_batch = None
    for step, (data, target) in enumerate(self.val_loader):
        # print('test step: {} total: {}'.format(step,max_test_iters))
        if one_batch == None:
          one_batch = data
        batchsize = data.shape[0]
        # print('get data',data.shape)
        target = target.cuda(non_blocking=True)
        #data, target = data.to(device), target.to(device)

        _, logits = self.model.forward(data, arch)#_with_architect

        prec1, prec5 = obtain_accuracy(logits.data, target.data, topk=(1, 5))
        base_top1.update  (prec1.item(), batchsize)
        base_top5.update  (prec5.item(), batchsize)

        del data, target, logits, prec1, prec5

    if self.lambda_t > 0.0:
      start_time = time.time()
      len_batch = min(len(one_batch), 50)
      for i in range(len_batch):
        _, _ = self.model.forward(one_batch[i:i+1, :, :, :], arch)
      end_time = time.time()
      time_per = (end_time - start_time) / len_batch
    else:
      time_per = 0.0

    #print('top1: {:.2f} top5: {:.2f}'.format(base_top1.avg * 100, base_top5.avg * 100))
    return base_top1.avg, base_top5.avg, time_per

  def cand_to_str(self, cand):
      return str(cand)

  def test_flops(self, cand):
      return 1

  def test_acc(self, cand):
      acc1, acc5, time_per = self.test_archi_acc(cand)
      if time_per > self.max_inference_time:
        return False
      elif time_per < self.want_inference_time:
        return acc1
      else:
        return acc1 - self.lambda_t * (time_per - self.want_inference_time)

  def is_legal(self, cand):
      cand_str = self.cand_to_str(cand)
      if cand_str not in self.perform_dict.keys():
          start_time = time.time()
          flops = self.test_flops(cand)
          self.perform_dict[cand_str] = self.test_acc(cand) if flops <= self.flops_limit else -1
          self.eva_time.update(time.time() - start_time,  1)
      if self.perform_dict[cand_str] == -1:
          return False
      else:
          return self.perform_dict[cand_str]

  def init_random(self):
      self.candidates = list()
      self.performances = list()
      while(len(self.candidates) < self.population_num):
          cand = self.model.module.uniform_sample()
          legal_result = self.is_legal(cand)
          if legal_result == False:
              continue
          self.candidates.append(cand)
          self.performances.append(legal_result)

  def search(self):
      self.eva_time = AverageMeter()
      init_start = time.time()
      self.init_random()
      self.logger.log('Initial_takes: %.2f'%(time.time()-init_start))

      epoch_start_time = time.time()
      epoch_time_meter = AverageMeter()
      bests_per_epoch = list()
      perform_trace = list()
      for i in range(self.max_epochs):
          self.performances = torch.Tensor(self.performances)
          top_k = torch.argsort(self.performances, descending=True)[:self.parent_num]

          if self.best_perf is None or self.performances[top_k[0]] > self.best_perf:
              self.best_cand = self.candidates[top_k[0]]
              self.best_perf = self.performances[top_k[0]]
          bests_per_epoch.append(self.best_cand)
          perform_trace.append(self.performances)
            
          self.parents = []
          for idx in top_k:
            self.parents.append(self.candidates[idx])
          self.candidates, self.performances = list(), list()
          self.eva_time = AverageMeter()
          self.get_mutation(self.population_num // 2)
          self.get_crossover()

          self.logger.log('*SEARCH* ' + time_string() + '||| Epoch: %2d finished, %3d models have been tested, best performance is %.2f'%(i, len(self.perform_dict.keys()), self.best_perf))
          self.logger.log(' - Best Cand: ' + str(self.best_cand))
          this_epoch_time = time.time() - epoch_start_time
          epoch_time_meter.update(this_epoch_time)
          epoch_start_time = time.time()
          self.logger.log('Time for Epoch %d : %.2fs'%(i, this_epoch_time))
          self.logger.log(' -- Evaluated %d models, with %.2f s in average'%(self.eva_time.count, self.eva_time.avg))

      self.logger.log('--------\nSearching Finished. Best Arch Found with Acc %.2f'%(self.best_perf))
      self.logger.log(str(self.best_cand))
      #torch.save(self.best_cand, self.save_dir+'/best_arch.pth')
      #torch.save(self.perform_dict, self.save_dir+'/perform_dict.pth')
      return bests_per_epoch, self.perform_dict, perform_trace

  def get_mutation(self, num):
    while len(self.candidates) < num:
      idx = random.randint(0, self.parent_num-1)
      new_cand = self.model.module.mutate(self.parents[idx])
      legal_result = self.is_legal(new_cand)
      if legal_result == False:
          continue
      self.candidates.append(new_cand)
      self.performances.append(legal_result)

  def get_crossover(self):
    while len(self.candidates) < self.population_num:
      idx1, idx2 = random.sample(range(self.parent_num), 2)
      new_cand = self.model.module.crossover(self.parents[idx1], self.parents[idx2])
      legal_result = self.is_legal(new_cand)
      if legal_result == False:
          continue
      self.candidates.append(new_cand)
      self.performances.append(legal_result)

class NASNetworkSPOS(nn.Module):

  def __init__(self, C: int, N: int, steps: int, multiplier: int, stem_multiplier: int,
               num_classes: int, search_space: List[Text], affine: bool, track_running_stats: bool, latency_file: str):
    super(NASNetworkSPOS, self).__init__()
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
      cell = SearchCell(search_space, steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, affine, track_running_stats, latency_file=latency_file)
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
    self.search_space = search_space

  def get_weights(self):
    xlist = list( self.stem.parameters() ) + list( self.cells.parameters() )
    xlist+= list( self.lastact.parameters() ) + list( self.global_pooling.parameters() )
    xlist+= list( self.classifier.parameters() )
    return xlist

  def uniform_sample(self):
    # sample operations and edges for normal cell
    index_normal = torch.randint(0, len(self.search_space), [self._steps * 4])
    index_reduce = torch.randint(0, len(self.search_space), [self._steps * 4])
    for i in range(self._steps):
      index_normal[4*i: 4*i+4: 2] = torch.multinomial(torch.ones(i+2), 2)
      index_reduce[4*i: 4*i+4: 2] = torch.multinomial(torch.ones(i+2), 2)

    return torch.cat([index_normal, index_reduce], axis = 0)

  def forward_with_architect(self, inputs, arch):
    s0 = s1 = self.stem(inputs)
    arch_normal, arch_reduce = arch[:self._steps * 4], arch[self._steps * 4:]
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell.forward_sample(s0, s1, arch_reduce if cell.reduction else arch_normal)
    out = self.lastact(s1)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return out, logits

  def forward(self, inputs, arch = None):
    if arch == None:
        arch = self.uniform_sample()
    return self.forward_with_architect(inputs, arch)

  def TrackRunningStats(self):
    self.apply(TrackRunningStats)

  def mutate(self, arch):
    arch = arch.clone().detach()
    for offset in [0, 4 * self._steps]:
      for in_id in [0, 2]:
        for node in range(self._steps):
          if random.random() < 0.1:
            arch[4 * node + in_id + offset] = torch.randint(0, 2+node, [1])
          if random.random() < 0.1:
            arch[4 * node + in_id + 1 + offset] = torch.randint(0, 8, [1])
    return arch

  def crossover(self, arch1, arch2):
    arch = arch1.clone().detach()
    split_idx = random.randint(1, len(arch)-1)
    arch[split_idx:] = arch2[split_idx:]
    return arch

  def getSearcher(self, model, train_loader, val_loader, logger, config):
    return Searcher(model, train_loader, val_loader, logger, config)