import argparse
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from ptflops import get_model_complexity_info

from lib.networks import TRANSNEXT_Cascaded
from trainer import trainer_synapse

parser = argparse.ArgumentParser()

parser.add_argument('--root_path', type=str,
                    default='./data/synapse/train_npz', help='root dir for data')
parser.add_argument('--volume_path', type=str,
                    default='./data/synapse/test_vol_h5', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=8, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='segmentation network learning rate')  # 0.001
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')  # 224
parser.add_argument('--alpha', '-a', type=float, default=32.0, help='alpha for dynamic kappa update')
parser.add_argument('--seed', type=int,
                    default=2222, help='random seed')

args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True

    args.exp = 'TRANSNEXT_Cascaded_' + dataset_name + str(args.img_size)
    snapshot_path = "model_pth/{}/{}".format(args.exp, 'TRANSNEXT_Cascaded')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    current_time = time.strftime("%H%M%S")
    print("The current time is", current_time)
    snapshot_path = snapshot_path + '_run' + current_time

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    net = TRANSNEXT_Cascaded(n_class=args.num_classes, img_size_s1=(args.img_size, args.img_size),
                             img_size_s2=(224, 224),
                             model_scale='small', decoder_aggregation='additive', interpolation='bilinear')

    print('Model %s created, param count: %d' %
          ('MERIT_Cascaded: ', sum([m.numel() for m in net.parameters()])))

    net = net.cuda()

    macs, params = get_model_complexity_info(net, (3, args.img_size, args.img_size), as_strings=True,
                                             print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    trainer = {'Synapse': trainer_synapse, }
    trainer[dataset_name](args, net, snapshot_path)
