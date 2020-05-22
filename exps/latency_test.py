import torch
import sys
import argparse
import time

from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, get_nas_search_loaders
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net, get_search_spaces
from models import OPS, SearchSpaceNames

def testOp(op, counts = 30):
  layer = op(C_in=36, C_out=36, stride=1, affine=False, track_running_stats=True)
  inputs = torch.randn(1, 36, 40, 40).cuda()
  all_time = 0.0
  for i in range(50):
      out = layer(inputs)
  for i in range(counts):
      start_time = time.time()
      out = layer(inputs)
      all_time += time.time() - start_time
  return all_time / counts


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )

  result_dict = dict()
  for iop in SearchSpaceNames[xargs.search_space_name]:
    result_dict[iop] = testOp(iop)

  for k, v in result_dict.items():
      print('%8s : %.3f'%(k,v))

if __name__ == "__main__":
  parser = argparse.ArgumentParser('Latency')
  parser.add_argument('--search_space_name',  type=str,   default='darts',   help='The search space name.')
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  args = parser.parse_args()
  main(args)