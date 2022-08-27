import json
import os

import torch
from torch import nn

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
#
# lower = torch.randn((23, 1))
# ang = torch.randn((64, 23))
# ex = torch.ones((64, 23), device='cuda')
#
# lower_repeat = lower.reshape((1, 23)).repeat((64, 1))
#
# loss_fn = nn.MSELoss()
# loss = loss_fn(lower_repeat, ang)
# print(loss)

def get_model_list(checkpoint_dir):
    models = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir)]
    if models is None:
        return None
    models.sort()
    latest_model = models[-1]
    return latest_model


checkpoint_dir = "../checkpoint"
model_name = get_model_list(checkpoint_dir)  # 获取最新的.pth参数文件路径
print(model_name)
# model.load_state_dict(torch.load(model_name))  # 载入模型参数
# 获取已经训练的模型的轮次
train_epoch = int(model_name.split("_")[1].split(".")[0])
print(train_epoch)

# ang = torch.randn((48,23))
# print(ang.shape[0])

# 训练集有32560不同的手部图片，每种包含4个版本，共130240张RGB图片
# 手部图片mask有32560张
# 对应的json文件中数据有多少？   32560还是130240  根据scale.json查看 有32560个数据
# 测试集有3960张图片 scale.json同样有3960个数据
# base_path = "E:\DataSet\FreiHAND_pub_v2_eval"
# scale_json_path = os.path.join(base_path, "evaluation_scale.json")
# print(scale_json_path)
# with open(scale_json_path, 'r') as fi:
#     scale = json.load(fi)
# print(len(scale))

# 计算相对于中指关节的其他关节位置
# joint = torch.randn((4, 4, 3))
# print(joint)
# middle = joint[:, 2, :].unsqueeze(1)
# print(middle)
# print(middle.shape)
# joint = joint - middle
# print(joint)
# print(joint.shape)

# 保存输出记录
# log_path = "../log/train_log.txt"
# log = open(log_path, "a")
# for i in range(10):
#     log.write("result:{}\n".format(i))

