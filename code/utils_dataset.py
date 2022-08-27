import json
import os
import time

import numpy as np
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d


def load_db_annotation(base_path, set_name=None):
    if set_name is None:
        # only training set annotations are released so this is a valid default choice
        set_name = 'training'

    print('Loading FreiHAND {} dataset index ...'.format(set_name))
    t = time.time()

    # assumed paths to data containers
    k_path = os.path.join(base_path, '%s_K.json' % set_name)
    mano_path = os.path.join(base_path, '%s_mano.json' % set_name)
    xyz_path = os.path.join(base_path, '%s_xyz.json' % set_name)

    # load if exist
    K_list = json_load(k_path)
    mano_list = json_load(mano_path)
    xyz_list = json_load(xyz_path)

    # should have all the same length
    assert len(K_list) == len(mano_list), 'Size mismatch.'
    assert len(K_list) == len(xyz_list), 'Size mismatch.'

    print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time() - t))
    return list(zip(K_list, mano_list, xyz_list))


def db_size(set_name):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        return 32560  # number of unique samples (they exists in multiple 'versions')
    elif set_name == 'evaluation':
        return 3960
    else:
        assert 0, 'Invalid choice.'


class sample_version:
    gs = 'gs'  # green screen
    hom = 'hom'  # homogenized
    sample = 'sample'  # auto colorization with sample points
    auto = 'auto'  # auto colorization without sample points: automatic color hallucination

    db_size = db_size('training')

    @classmethod
    def valid_options(cls):
        return [cls.gs, cls.hom, cls.sample, cls.auto]

    @classmethod
    def check_valid(cls, version):
        msg = 'Invalid choice: "%s" (must be in %s)' % (version, cls.valid_options())
        assert version in cls.valid_options(), msg

    @classmethod
    def map_id(cls, id, version):
        cls.check_valid(version)
        return id + cls.db_size * cls.valid_options().index(version)


def read_img(idx, base_path, set_name, version=None):
    if version is None:
        version = sample_version.gs

    if set_name == 'evaluation':
        assert version == sample_version.gs, 'This the only valid choice for samples from the evaluation split.'

    img_rgb_path = os.path.join(base_path, set_name, 'rgb',
                                '%08d.jpg' % sample_version.map_id(idx, version))
    img = Image.open(img_rgb_path).convert('RGB')
    to_tensor = torchvision.transforms.ToTensor()
    return to_tensor(img)


def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]


# class FreiHandTrainSet(Dataset):
#
#     def __init__(self, root_dir):
#         self.root_dir = root_dir
#         self.img_path = os.path.join(self.root_dir, "training", "rgb")
#         self.data_anno = load_db_annotation(self.root_dir, "training")
#
#     def __getitem__(self, idx):
#         img = read_img(idx, self.root_dir, "training")
#
#         K, mano, xyz = self.data_anno[idx]
#         K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
#         uv = projectPoints(xyz, K)  # 21 keypoint 2d [21, 2]
#         return img, xyz, uv
#
#     def __len__(self):
#         return db_size('training')

class FreiHandSet(Dataset):

    def __init__(self, root_dir, split, version):
        # root_dir 表示数据集的根目录 取值 FreiHAND_pub_v2 or FreiHAND_pub_v2_eval
        # split表示训练集还是测试集  取值 training or evaluation
        self.root_dir = root_dir
        self.split = split
        self.version = version
        # self.img_path = os.path.join(self.root_dir, split, "rgb")
        self.data_anno = load_db_annotation(self.root_dir, split)

    def __getitem__(self, idx):
        img = read_img(idx, self.root_dir, self.split, self.version)

        K, mano, xyz = self.data_anno[idx]
        K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
        uv = projectPoints(xyz, K)  # 21 keypoint 2d [21, 2]
        return img, xyz, uv

    def __len__(self):
        return db_size(self.split)

#
# def get_data_loader(train_root_dir, test_root_dir, batch_size, num_workers):
#     trainDataset = FreiHandTrainSet(train_root_dir)
#     testDataset = FreiHandTestSet(test_root_dir)
#     trainloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     testloader = DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#
#     return trainloader, testloader


if __name__ == '__main__':
    train_root_dir = "E:\DataSet\FreiHAND_pub_v2"
    test_root_dir = "E:\DataSet\FreiHAND_pub_v2_eval"
    train = FreiHandSet(train_root_dir, split="training")
    print(len(train))
    # img, joint, keypoint = train[0]
    # print(joint * 1000)
    # print(keypoint)
    # test = FreiHandSet(test_root_dir, split="evaluation")
    # print(len(test))
    # img, joint, keypoint = test[0]
    # print(joint)