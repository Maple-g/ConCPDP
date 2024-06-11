import os
import random

import numpy as np
import pickle

import torch


def round_self(array):
    result = list()
    for a in array:
        if a > 0.5:
            result.append(1)
        else:
            result.append(0)
    return result

def resam(ts, ts1, ts2, ts3, l):
    ts = torch.from_numpy(ts)
    ts1 = torch.from_numpy(ts1)
    ts2 = torch.from_numpy(ts2)
    ts3 = torch.from_numpy(ts3)
    l = torch.from_numpy(l)
    n = ts.shape[0]
    while ts.shape[0] < 100:
        rand = random.randint(0, n-1)
        ts = torch.cat((ts, ts[rand].unsqueeze(0)), 0)
        ts1 = torch.cat((ts1, ts1[rand].unsqueeze(0)), 0)
        ts2 = torch.cat((ts2, ts2[rand].unsqueeze(0)), 0)
        ts3 = torch.cat((ts3, ts3[rand].unsqueeze(0)), 0)
        l = torch.cat((l, l[rand].unsqueeze(0)), 0)
    ts = ts.detach().numpy()
    ts1 = ts1.detach().numpy()
    ts2 = ts2.detach().numpy()
    ts3 = ts3.detach().numpy()
    l = l.detach().numpy()
    return ts, ts1, ts2, ts3, l

def generate_dir_recursive(path):
    import os
    path = path.strip()
    path = path.rstrip('/')
    phases = path.split('/')
    path = ''
    for i in range(len(phases)):
        path = path + phases[i] + '/'
        if not os.path.exists(path):
            os.mkdir(path)


def dump_data(path, obj):     #将python中的信息（元组，列表，字典等）进行序列化，然后保存为文件形式。不过该文件一般不可读
    with open(path, 'wb') as file_obj:
        pickle.dump(obj, file_obj)


def load_data(path):          #将序列化后的文件解析为原来的python对象
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as file_obj:   #rb表示文件只读
        return pickle.load(file_obj)


def generate_weight_for_instance(label):
    a = 0
    b = 0
    label = np.array(label)
    label = label.reshape(-1)
    for i in label:
        if i == 0:
            a = a + 1
        else:
            b = b + 1
    s = a + b
    a = a * 1.0 / s
    b = b * 1.0 / s
    result = []
    for i in label:
        if i == 0:
            result.append(b)
        else:
            result.append(a)
    return np.array(result)


def generate_weight_for_class(label):
    a = 0
    b = 0
    label = np.array(label)
    label = label.reshape(-1)
    for i in label:
        if i == 0:
            a = a + 1
        else:
            b = b + 1
    s = a + b
    a = a * 1.0 / s
    b = b * 1.0 / s
    result = [b, a]
    return np.array(result)


def imbalance_process(x_, x_handcraft, label, processor, processor_default):
    if processor is None:
        return x_, x_handcraft, label

    _len_ast = len(x_[0])
    _len_label = len(label)

    _x = np.hstack((x_, x_handcraft))
    _y = label.reshape(_len_label)
    try:
        _x_resampled, _y_resampled = processor.fit_sample(_x, _y)
    except ValueError:
        _x_resampled, _y_resampled = processor_default.fit_sample(_x, _y)

    _ast, _handcraft = np.split(_x_resampled, [_len_ast], axis=1)
    _label = _y_resampled.reshape((len(_y_resampled), 1))

    print("after imbalance process:" + str(len(_y_resampled)))
    _count = 0
    for i in _y_resampled:
        if i == 0:
            _count += 1
    print("0 count is:" + str(_count))

    return _ast, _handcraft, _label


def imbalance_process_for_aug(x_, x_aug1, x_aug2, t_, t_aug1, t_aug2, x_handcraft, t_handcraft, label, t_label, processor, processor_default):
    if processor is None:
        return x_, x_aug1, x_aug2, x_handcraft, label, t_, t_aug1, t_aug2, t_handcraft, t_label

    _len_ast = len(x_[0])
    _len_aug = len(x_aug1[0])
    _len_label = len(label)


    _x = np.hstack((x_, x_aug1, x_aug2, x_handcraft))
    _y = label.reshape(_len_label)

    try:
        _x_resampled, _y_resampled = processor.fit_sample(_x, _y)
    except ValueError:
        _x_resampled, _y_resampled = processor_default.fit_sample(_x, _y)

    _ast, _ast_aug1, _ast_aug2, _handcraft = np.split(_x_resampled, [_len_ast, (_len_ast+_len_aug), (_len_ast+_len_aug*2)], axis=1)
    _label = _y_resampled.reshape((len(_y_resampled), 1))


    print("after imbalance process:" + str(len(_y_resampled)))
    _count = 0
    for i in _y_resampled:
        if i == 0:
            _count += 1
    print("0 count is:" + str(_count))

    return _ast, _ast_aug1, _ast_aug2, _handcraft, _label, t_, t_aug1, t_aug2, t_handcraft, t_label


def count_binary(label):
    _count = 0
    for i in range(len(label)):
        if label[i][0] == 0:
            _count += 1
    print("0:" + str(_count))
    print("1:" + str(len(label) - _count))