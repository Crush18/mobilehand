import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils_dataset import *
from utils_neural_network import HMR
from utils_zimeval import *

# User selection
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--cuda', '-c', action='store_true', help='Use GPU')
parser.add_argument('--data', '-d', default='freihand', help='stb / freihand')
parser.add_argument('--mode', '-m', default='image', help='image / video / camera')
arg = parser.parse_args()

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load testdataset
batch_size = 80  # HMR模型的最大batch是80
num_workers = 0

test_root_dir = "E:\DataSet\FreiHAND_pub_v2"
test = FreiHandSet(test_root_dir, split="training")
testloader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Load neural network
model = HMR(arg)
# model.load_state_dict(torch.load('../model/hmr_model_' + arg.data + '_auc.pth'))
model.load_state_dict(torch.load("../checkpoint/model_101.pth"))
model.to(device)

# 损失函数
loss_mse = nn.MSELoss()

# eval
total_test_auc = 0
total_test_loss = 0
total_test_step = 0
model.eval()
with torch.no_grad():
    for data in testloader:
        imgs, gt_joints, gt_keypoints = data
        # 真实标注中   3djoint 扩大1000倍数  并改为相对于中指MCP的位置
        gt_joints = gt_joints * 1000
        gt_joints = gt_joints - gt_joints[:, 9, :].unsqueeze(1)  # 相对中指MCP的位置
        imgs = imgs.to(device)  # [bs, 3, 224, 224]
        gt_joints = gt_joints.to(device)  # [bs, 21, 3]
        gt_keypoints = gt_keypoints.to(device)  # [bs, 21, 2]

        # 模型计算结果
        keypt, joint, vert, ang, params = model(imgs,
                                                evaluation=True)  # [bs,21,2], [bs,21,3], [bs,778,3], [bs,23] [bs, 39]
        shape = params[:, 6:16].contiguous()  # shape在params中的位置 [bs, 10]

        # 计算损失
        loss_2d = loss_mse(gt_keypoints, keypt)
        loss_3d = loss_mse(gt_joints, joint)

        # 关节角度范围约束
        alim = model.mano.alim_.to(device)  # [23, 2]
        lower_bound = alim[:, 0].reshape((1, 23)).repeat((ang.shape[0], 1))  # [23, 1]   ang [bs, 23]
        upper_bound = alim[:, 1].reshape((1, 23)).repeat((ang.shape[0], 1))  # [bs, 23]
        ang_constraint = torch.sum(torch.max(torch.zeros((ang.shape[0], 23), device='cuda'), lower_bound - ang)
                                   + torch.max(torch.zeros((ang.shape[0], 23), device='cuda'), ang - upper_bound))
        loss_reg = loss_mse(shape, torch.zeros((shape.shape[0], 10), device='cuda')) + ang_constraint
        total_loss = 100 * loss_2d + 100 * loss_3d + 1000 * loss_reg

        total_test_loss = total_test_loss + total_loss.item()

        # 计算3D PCK 和 AUC  要求joint的gt和pred在同一个数量级上
        evaluator = EvalUtil()
        start_time = time.time()
        batch_test_auc = 0
        for i in range(joint.shape[0]):
            gt_joint_item = gt_joints[i]
            pred_joint_item = joint[i]
            evaluator.feed(gt_joint_item, pred_joint_item)
            result = evaluator.get_measures(0, 50, 20)
            _1, _2, _3, auc_3d, pck_curve_3d, _ = result
            print("step:{}, single test auc:{}".format(total_test_step, auc_3d))
            batch_test_auc  = batch_test_auc + auc_3d
            total_test_auc = total_test_auc + auc_3d
        end_time = time.time()
        print("step:{}, batch sum time:{}, batch test auc:{}".format(total_test_step, end_time-start_time, batch_test_auc / joint.shape[0]))
        total_test_step = total_test_step + 1

    print("total_test_loss:{}".format(total_test_loss))
    print("total_test_auc:{}".format(total_test_auc/ len(test)))
