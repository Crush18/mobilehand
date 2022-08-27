import argparse
import os

import torch
from torch import nn

from utils_dataset import *
from utils_neural_network import HMR
from utils_zimeval import *

parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--cuda', '-c', action='store_true', help='Use GPU')
parser.add_argument('--data', '-d', default='freihand', help='stb / freihand')
parser.add_argument('--mode', '-m', default='image', help='image / video / camera')
arg = parser.parse_args()

# 定义训练设备  训练时默认使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# dataloader 获取每个样本的rgb图像和标注-3djoint和2dkeypoint
# 数据集存放在项目文件的同级目录
train_root_dir = "E:\DataSet\FreiHAND_pub_v2"
test_root_dir = "E:\DataSet\FreiHAND_pub_v2_eval"
# train_root_dir = "../../FreiHAND_pub_v2"
# test_root_dir = "../../FreiHAND_pub_v2_eval"

batch_size = 20   # HMR模型的最大batch是80
num_workers = 0

# 准备数据集并加载数据
train = FreiHandSet(train_root_dir, split="training")
trainloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test = FreiHandSet(test_root_dir, split="evaluation")
testloader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# 创建模型
model = HMR(arg)
model.to(device)

# 损失函数
loss_mse = nn.MSELoss()

# 优化器
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)

# 训练
total_train_step = 0
total_test_step = 0
epoch = 400
train_epoch = 0

# test pck 测试
total_test_auc = 0
evaluator = EvalUtil()


# 写入tensorboard
# writer = SummaryWriter("../logs_model")
# 写入log日志  追加写入
# log = open("../log/train_log.txt", "a")


# 从之前训练的模型中载入参数

def get_model_list(checkpoint_dir):
    models = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)]
    if len(models) == 0:
        return None
    models.sort()
    latest_model = models[-1]
    return latest_model


checkpoint_dir = "../checkpoint"
model_name = get_model_list(checkpoint_dir)  # 获取最新的.pth参数文件路径
if model_name is None:
    train_epoch = 0
else:
    model.load_state_dict(torch.load(model_name))  # 载入模型参数
    train_epoch = int(model_name.split("_")[1].split(".")[0])  # 获取已经训练的模型的轮次

while True:
    print("----------------------第 {} 轮训练开始--------------------".format(train_epoch + 1))
    # log.write("----------------------第 {} 轮训练开始--------------------\n".format(train_epoch))

    # 训练集上训练
    model.train()
    for data in trainloader:
        imgs, gt_joints, gt_keypoints = data
        # 真实标注中   3djoint 扩大1000倍数  并改为相对于中指MCP的位置
        gt_joints = gt_joints * 1000
        gt_joints = gt_joints - gt_joints[:, 9, :].unsqueeze(1)  # 相对中指MCP的位置
        imgs = imgs.to(device)  # [bs, 3, 224, 224]
        gt_joints = gt_joints.to(device)  # [bs, 21, 3]
        gt_keypoints = gt_keypoints.to(device)  # [bs, 21, 2]

        # 模型计算结果
        keypt, joint, vert, ang, params = model(imgs,
                                                evaluation=False)  # [bs,21,2], [bs,21,3], [bs,778,3], [bs,23] [bs, 39]
        shape = params[-1][:, 6:16].contiguous()  # shape在params中的位置 [bs, 10]

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

        # 计算3D PCK 和 AUC  要求joint的gt和pred在同一个数量级上   验证功能使用
        for i in range(joint.shape[0]):
            gt_joint_item = gt_joints[i]
            pred_joint_item = joint[i]
            evaluator.feed(gt_joint_item, pred_joint_item)
            result = evaluator.get_measures(0, 50, 20)
            _1, _2, _3, auc_3d, pck_curve_3d, _ = result
            total_test_auc = total_test_auc + auc_3d

        # 优化器优化模型
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        schedule.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss:{}".format(total_train_step, total_loss))
            # log.write("训练次数：{}, Loss:{}\n".format(total_train_step, total_loss))
            # writer.add_scalar("train", total_loss, total_train_step)

    # 测试集上测试  不需要计算梯度
    model.eval()
    total_test_loss = 0
    total_test_auc = 0


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
            for i in range(joint.shape[0]):
                gt_joint_item = gt_joints[i]
                pred_joint_item = joint[i]
                evaluator.feed(gt_joint_item, pred_joint_item)
                result = evaluator.get_measures(0, 50, 20)
                _1, _2, _3, auc_3d, pck_curve_3d, _ = result
                total_test_auc = total_test_auc + auc_3d

        print("整体测试集上Loss:{}".format(total_test_loss))
        # log.write("整体测试集上Loss:{}\n".format(total_test_loss))
        print("整体测试集上AUC:{}".format(total_test_auc / len(test)))
        # log.write("整体测试集上AUC:{}\n".format(total_test_auc / len(test)))

    # 每轮训练后保存模型
    torch.save(model.state_dict(), "../checkpoint/model_{}.pth".format(train_epoch + 1))
    print("模型已经保存")
    # log.write("模型已经保存\n")

    train_epoch = train_epoch + 1
    if train_epoch > epoch:
        break

# writer.close()
