import torch
import sys
import argparse
import time

from cell_operations import OPS, SearchSpaceNames

def testOp(op, channel, size, stride, counts = 1000):
  layer = OPS[op](C_in=channel, C_out=channel, stride=stride, affine=False, track_running_stats=True).cuda()
  size = size * stride
  inputs = torch.randn(1, channel, size, size).cuda()
  print('-- Input(1x%dx%dx%d)'%(channel, size, size))
  all_time = 0.0
  for i in range(100):
      out = layer(inputs)
  start_time = time.time()
  for i in range(counts):
      out = layer(inputs)
  all_time = time.time() - start_time
  return all_time


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )

  '''
  result_dict = dict()
  for iop in SearchSpaceNames[xargs.search_space_name]:
    print("Testing for %s ..."%(iop))
    for c, s in zip([36, 72, 144], [32, 16, 8]):
      k = '%s_%d_%d_%d'%(iop, c, s, 1)
      result_dict[k] = testOp(iop, c, s, 1, xargs.counts)
      k = '%s_%d_%d_%d'%(iop, c, s, 2)
      result_dict[k] = testOp(iop, c, s, 2, xargs.counts)

  for k, v in result_dict.items():
      print('%18s : %.3f'%(k,v))
      
  save_dict = dict()
  save_dict['space'] = xargs.search_space_name
  save_dict['counts'] = xargs.counts
  save_dict['latency'] = result_dict

  torch.save(save_dict, "latency_large.pth")
  '''

  # v2
  result_dict = dict()
  all_ops = SearchSpaceNames[xargs.search_space_name]
  for c, s in zip([16, 32, 64], [32, 16, 8]):
    v_1 = torch.zeros((len(all_ops)))
    v_2 = torch.zeros((len(all_ops)))
    for i,iop in enumerate(all_ops):
      print("Testing for %s ..."%(iop))
      v_1[i] = testOp(iop, c, s, 1, xargs.counts)
      v_2[i] = testOp(iop, c, s, 2, xargs.counts)
    result_dict['%d_%d_%d'%(c, s, 1)] = v_1
    result_dict['%d_%d_%d'%(c, s, 2)] = v_2

  for k, v in result_dict.items():
      print(k, v)
      
  save_dict = dict()
  save_dict['space'] = xargs.search_space_name
  save_dict['counts'] = xargs.counts
  save_dict['latency'] = result_dict

  torch.save(save_dict, "darts_small_vector.pth")


if __name__ == "__main__":
  parser = argparse.ArgumentParser('Latency')
  parser.add_argument('--search_space_name',  type=str,   default='darts',   help='The search space name.')
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--counts',            type=int,   default=1000,    help='number of data loading workers (default: 2)')
  args = parser.parse_args()
  main(args)