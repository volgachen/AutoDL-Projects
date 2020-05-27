##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
###########################################################################
# Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019 #
###########################################################################
import sys, time, random, argparse
from copy import deepcopy
import torch
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config
from datasets     import get_datasets, get_nas_search_loaders
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net, get_search_spaces
from nas_201_api  import NASBench201API as API


def train_shared_cnn(xloader, shared_cnn, criterion, scheduler, optimizer, print_freq, logger, config, start_epoch):
  # start training
  start_time, epoch_time, total_epoch = time.time(), AverageMeter(), config.epochs + config.warmup
  for epoch in range(start_epoch, total_epoch):
    scheduler.update(epoch, 0.0)
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_epoch-epoch), True) )
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
    logger.log('\n[Traing the {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, min(scheduler.get_lr())))

    data_time, batch_time = AverageMeter(), AverageMeter()
    losses, top1s, top5s, xend = AverageMeter(), AverageMeter(), AverageMeter(), time.time()
    
    shared_cnn.train()

    for step, (inputs, targets) in enumerate(xloader):
      scheduler.update(None, 1.0 * step / len(xloader))
      targets = targets.cuda(non_blocking=True)
      # measure data loading time
      data_time.update(time.time() - xend)

      optimizer.zero_grad()
      _, logits = shared_cnn(inputs)
      loss      = criterion(logits, targets)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(shared_cnn.parameters(), 5)
      optimizer.step()
      # record
      prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
      losses.update(loss.item(),  inputs.size(0))
      top1s.update (prec1.item(), inputs.size(0))
      top5s.update (prec5.item(), inputs.size(0))

      # measure elapsed time
      batch_time.update(time.time() - xend)
      xend = time.time()

      if step % print_freq == 0 or step + 1 == len(xloader):
        Sstr = '*Train-Shared-CNN* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
        Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
        Wstr = '[Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=losses, top1=top1s, top5=top5s)
        logger.log(Sstr + ' ' + Tstr + ' ' + Wstr)
        
    cnn_loss, cnn_top1, cnn_top5 = losses.avg, top1s.avg, top5s.avg
    logger.log('[{:}] shared-cnn : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, cnn_loss, cnn_top1, cnn_top5))
    epoch_time.update(time.time() - start_time)
    start_time = time.time()
  return 


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  #config_path = 'configs/nas-benchmark/algos/GDAS.config'
  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  search_loader, train_loader, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/nas-benchmark/', config.batch_size, xargs.workers)
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  search_space = get_search_spaces('cell', xargs.search_space_name)
  if xargs.model_config is None:
    model_config = dict2config({'name': 'GDAS', 'C': xargs.channel, 'N': xargs.num_cells,
                                'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                'space'    : search_space,
                                'affine'   : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  else:
    model_config = load_config(xargs.model_config, {'num_classes': class_num, 'space'    : search_space,
                                                    'affine'     : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  search_model = get_cell_based_tiny_net(model_config)
  logger.log('search-model :\n{:}'.format(search_model))
  logger.log('model-config : {:}'.format(model_config))
  
  w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)
  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))
  flop, param  = get_model_infos(search_model, xshape)
  logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
  logger.log('search-space [{:} ops] : {:}'.format(len(search_space), search_space))
  if xargs.arch_nas_dataset is None:
    api = None
  else:
    api = API(xargs.arch_nas_dataset)
  logger.log('{:} create API = {:} done'.format(time_string(), api))

  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()

  if False:#last_info.exists(): # automatically resume from previous checkpoint
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info   = torch.load(last_info)
    start_epoch = last_info['epoch']
    checkpoint  = torch.load(last_info['last_checkpoint'])
    search_model.load_state_dict( checkpoint['search_model'] )
    w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
    w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
    logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch, valid_accuracies = 0, {'best': -1}

  if len(xargs.supernet_path)>0:
    saved_info = torch.load(xargs.supernet_path)
    assert saved_info['epoch'] == 'finished', "Epoch is not finished in this file"
    search_model.load_state_dict(saved_info['search_model'])
  else:
    # start training supernet
    start_time = time.time()
    train_shared_cnn(train_loader, network, criterion, w_scheduler, w_optimizer, xargs.print_freq, logger, config, start_epoch)
    logger.log('Supernet trained. Time-cost = {:.1f} s'.format(time.time()-start_time))
    # save supernetweight
    save_path = save_checkpoint({'epoch' : 'finished',#epoch + 1,
                'args'  : deepcopy(xargs),
                'search_model': search_model.state_dict(),
                'w_optimizer' : w_optimizer.state_dict(),
                'w_scheduler' : w_scheduler.state_dict()},
                model_base_path, logger)
    last_info = save_checkpoint({
          'epoch': 'finished',#epoch + 1,
          'args' : deepcopy(args),
          'last_checkpoint': save_path,
          }, logger.path('info'), logger)

  search_start_time = time.time()
  searcher = search_model.getSearcher(network, train_loader, valid_loader, logger, config)
  best_cands, performance_dict, performance_trace = searcher.search()
  logger.log('Architect Searched. Time-cost = {:.1f} s'.format(time.time()-search_start_time))
  search_result = save_checkpoint({
        'epoch': 'finished',#epoch + 1,
        'args' : deepcopy(args),
        'genotypes': best_cands,
        'performance_dict': performance_dict,
        'performance_trace': performance_trace
        }, model_best_path, logger)
  

  logger.close()
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser("SPOS")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--track_running_stats',type=int,   choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--config_path',        type=str,   help='The path of the configuration.')
  parser.add_argument('--model_config',       type=str,   help='The path of the model configuration. When this arg is set, it will cover max_nodes / channels / num_cells.')
  # load_from
  parser.add_argument('--supernet_path',      type=str,   default='',   help='load supernet from file')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
