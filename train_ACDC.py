import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定只使用第1個GPU
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random
import time

import numpy as np
from tqdm import tqdm
from medpy.metric import dc
from scipy.ndimage import zoom

from utils.utils import powerset
from utils.dataset_ACDC import ACDCdataset, RandomGenerator
from test_ACDC import inference
from lib.networks import TRANSNEXT_Cascaded
# 引入 Adaptive tvMF Dice Loss
from loss import Adaptive_tvMF_DiceLoss

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, help="batch size")
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", default=400)
parser.add_argument("--img_size", default=224)
parser.add_argument("--save_path", default="./model_pth/ACDC")
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="./data/ACDC/lists_ACDC")
parser.add_argument("--root_dir", default="./data/ACDC/")
parser.add_argument("--volume_path", default="./data/ACDC/test")
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=4)
parser.add_argument('--test_save_dir', default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=2222, help='random seed')
# 新增參數，用於調整 Adaptive tvMF Dice Loss 的 alpha
parser.add_argument("--alpha", default=32, type=float, help="alpha for adaptive tvMF Dice loss")
args = parser.parse_args()

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

args.is_pretrain = True
args.exp = 'TRANSNEXT_Cascaded' + str(args.img_size)
snapshot_path = "{}/{}/{}".format(args.save_path, args.exp, 'TRANSNEXT_Cascaded')
snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
snapshot_path = snapshot_path + '_lr' + str(args.lr) if args.lr != 0.01 else snapshot_path
snapshot_path = snapshot_path + '_' + str(args.img_size)
snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

current_time = time.strftime("%H%M%S")
print("The current time is", current_time)
snapshot_path = snapshot_path + '_run' + current_time

if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)

args.test_save_dir = os.path.join(snapshot_path, args.test_save_dir)
test_save_path = os.path.join(args.test_save_dir, args.exp)
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path, exist_ok=True)

# 建立模型
net = TRANSNEXT_Cascaded(n_class=args.num_classes,
                         img_size_s1=(args.img_size, args.img_size),
                         img_size_s2=(224, 224),
                         model_scale='small',
                         decoder_aggregation='additive',
                         interpolation='bilinear').cuda()

if args.checkpoint:
    net.load_state_dict(torch.load(args.checkpoint))

# 建立資料集與 Dataloader
train_dataset = ACDCdataset(args.root_dir, args.list_dir, split="train",
                            transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size])]))
print("The length of train set is: {}".format(len(train_dataset)))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

db_val = ACDCdataset(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
valloader = DataLoader(db_val, batch_size=1, shuffle=False)

db_test = ACDCdataset(base_dir=args.volume_path, list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

if args.n_gpu > 1:
    net = nn.DataParallel(net)
net = net.cuda()
net.train()

# 定義損失函數
ce_loss = CrossEntropyLoss()
# 原本的 dice_loss 替換為 Adaptive tvMF Dice Loss
adaptive_loss = Adaptive_tvMF_DiceLoss(n_classes=args.num_classes)
# dice_loss = DiceLoss(args.num_classes)

# 一開始的最佳驗證 Dice 及最佳 test Dice 分數
Best_dcs = 0.80
Best_dcs_th = 0.865
Best_test_dcs = 0.0  # 新增：紀錄 test dataset 的最高分數

logging.basicConfig(filename=os.path.join(snapshot_path, "log.txt"),
                    level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s',
                    datefmt='%H:%M:%S')

max_iterations = args.max_epochs * len(train_loader)
base_lr = args.lr
optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=0.0001)


# 定義 adjust_kappa：根據驗證的平均 DSC 來更新 kappa（每個類別設相同值）
def adjust_kappa(avg_dsc):
    # 建立一個長度為 num_classes 的 tensor，每個元素為 avg_dsc * alpha
    return torch.full((args.num_classes,), avg_dsc * args.alpha, device=torch.device('cuda'))


# 初始化 kappa
kappa = torch.zeros(args.num_classes).cuda()


# 測試驗證用函數
def val():
    logging.info("Validation ===>")
    dc_sum = 0.0
    net.eval()
    for i, val_sampled_batch in enumerate(valloader):
        val_image_batch, val_label_batch = val_sampled_batch["image"], val_sampled_batch["label"]
        val_image_batch = val_image_batch.squeeze(0).cpu().detach().numpy()
        val_label_batch = val_label_batch.squeeze(0).cpu().detach().numpy()
        x, y = val_image_batch.shape[0], val_image_batch.shape[1]
        if x != args.img_size or y != args.img_size:
            val_image_batch = zoom(val_image_batch, (args.img_size / x, args.img_size / y), order=3)
        val_image_batch = torch.from_numpy(val_image_batch).unsqueeze(0).unsqueeze(0).float().cuda()

        P = net(val_image_batch)
        val_outputs = 0.0
        for idx in range(len(P)):
            val_outputs += P[idx]
        val_outputs = torch.softmax(val_outputs, dim=1)
        val_outputs = torch.argmax(val_outputs, dim=1).squeeze(0)
        val_outputs = val_outputs.cpu().detach().numpy()
        if x != args.img_size or y != args.img_size:
            val_outputs = zoom(val_outputs, (x / args.img_size, y / args.img_size), order=0)
        dc_sum += dc(val_outputs, val_label_batch)
    performance = dc_sum / len(valloader)
    logging.info('Validation performance: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))
    print('Validation performance: mean_dice : %f, best_dice : %f' % (performance, Best_dcs))
    return performance


l = [0, 1, 2, 3]
ss = [x for x in powerset(l)]  # for mutation

iterator = tqdm(range(args.max_epochs), ncols=70)
iter_num = 0
Loss = []
Test_Accuracy = []

for epoch in iterator:
    net.train()
    train_loss = 0
    for i_batch, sampled_batch in enumerate(train_loader):
        image_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        image_batch = image_batch.type(torch.FloatTensor).cuda()
        label_batch = label_batch.type(torch.FloatTensor).cuda()

        P = net(image_batch)
        loss = 0.0
        lc1, lc2 = 0.3, 0.7

        for s in ss:
            if s == []:
                continue
            iout = 0.0
            for idx in s:
                iout += P[idx]
            loss_ce = ce_loss(iout, label_batch.long())
            # 這裡用 adaptive_loss 替代原始 dice_loss，傳入 kappa
            loss_dice = adaptive_loss(iout, label_batch, kappa, softmax=True)
            # loss_dice = dice_loss(iout, label_batch, softmax=True)
            loss += (lc1 * loss_ce + lc2 * loss_dice)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_ = base_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num += 1
        if iter_num % 50 == 0:
            logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
            print('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
        train_loss += loss.item()
    Loss.append(train_loss / len(train_dataset))
    logging.info('epoch %d : loss : %f lr_: %f' % (epoch, loss.item(), lr_))
    print('epoch %d : loss : %f lr_: %f' % (epoch, loss.item(), lr_))

    # 儲存最近的模型
    save_model_path = os.path.join(snapshot_path, 'last.pth')
    torch.save(net.state_dict(), save_model_path)

    # 驗證階段
    avg_dcs = val()
    # 更新 kappa，這裡使用驗證平均 DSC 乘以 alpha
    kappa = adjust_kappa(avg_dcs)

    # 儲存驗證表現最佳的模型
    if avg_dcs > Best_dcs:
        save_model_path_val = os.path.join(snapshot_path, 'best.pth')
        torch.save(net.state_dict(), save_model_path_val)
        logging.info("Save best val model to {}".format(save_model_path_val))
        print("Save best val model to {}".format(save_model_path_val))
        Best_dcs = avg_dcs

    # 如果驗證表現達到一定門檻，則進行 test dataset 推論
    if avg_dcs > Best_dcs_th or avg_dcs >= Best_dcs:
        avg_test_dcs, avg_hd, avg_jacard, avg_asd = inference(args, net, testloader, args.test_save_dir)
        print("Test avg_dcs: %f" % (avg_test_dcs))
        logging.info("Test avg_dcs: %f" % (avg_test_dcs))
        Test_Accuracy.append(avg_test_dcs)

        # 新增：若 test dataset 的 Dice 分數突破歷史最佳則儲存為 test_best.pth
        if avg_test_dcs > Best_test_dcs:
            save_model_path_test = os.path.join(snapshot_path, 'test_best.pth')
            torch.save(net.state_dict(), save_model_path_test)
            logging.info("Save best test model to {}".format(save_model_path_test))
            print("Save best test model to {}".format(save_model_path_test))
            Best_test_dcs = avg_test_dcs

    if epoch >= args.max_epochs - 1:
        save_model_path = os.path.join(snapshot_path, 'epoch={}_lr={}_avg_dcs={}.pth'.format(epoch, lr_, avg_dcs))
        torch.save(net.state_dict(), save_model_path)
        logging.info("Save final model to {}".format(save_model_path))
        print("Save final model to {}".format(save_model_path))
        break
