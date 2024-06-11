import random
import sklearn.neighbors._base
import sys
import os
import copy
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = '0/'

sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

from tool.imblearn.over_sampling import RandomOverSampler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.svm import SVC
import datetime
import itertools
import torch.optim as optim
import torch
import models.LSTM as LSTM
from MMD import mmd_loss

import models.Classifier as cls
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD
from ParsingSource import *
from Tools import *
from models import ContrastiveLossELI5

count = 0

# -------Auxiliary method--------Start

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class MyDataset(Dataset):
    def __init__(self, dataset1, dataset2, dataset3):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3

    def __getitem__(self, index):
        x1 = self.dataset1[index]
        x2 = self.dataset2[index]
        x3 = self.dataset3[index]
        return x1, x2, x3

    def __len__(self):
        return len(self.dataset1)


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# Parse arguments
parser = argparse.ArgumentParser(description='Train and evaluate models on code property prediction')
parser.add_argument('--regenerate', type=bool, default=False, help='Regenerate data flag')
parser.add_argument('--dump_data_path', type=str, default=None, help='Path to dump data')
parser.add_argument('--loop_size', type=int, default=None, help='Size of the loop for training')
parser.add_argument('--maxlen', type=int, default=None, help='Maximum length of sequences')
parser.add_argument('--opt_node', type=int, nargs='+', help='List of options for number of nodes')
parser.add_argument('--opt_batchsize', type=int, nargs='+', help='List of options for batch sizes')
parser.add_argument('--opt_epoch', type=int, nargs='+', help='List of options for number of epochs')
parser.add_argument('--opt_learning_rate', type=float, nargs='+', help='List of options for learning rates')
args = parser.parse_args()

DEVICE = torch.device('cuda')
root_path_source = 'data/projects/'
package_heads = ['org', 'gnu', 'bsh', 'javax', 'com']

start_time = datetime.datetime.now()
start_time_str = start_time.strftime('%Y-%m-%d_%H.%M.%S')

path_train_and_test = []
with open('data/pairs.txt', 'r') as file_obj:
    for line in file_obj.readlines():
        line = line.strip('\n').strip(' ')
        path_train_and_test.append(line.split(','))

for path in path_train_and_test:
    path_train_source = root_path_source + path[0]
    path_test_source = root_path_source + path[1]

    print(path[0] + "===" + path[1])
    train_project_name = path_train_source.split('/')[2]
    test_project_name = path_test_source.split('/')[2]
    path_train_and_test_set = args.dump_data_path + train_project_name + '_to_' + test_project_name

    if os.path.exists(path_train_and_test_set) and not args.regenerate:
        obj = load_data(path_train_and_test_set)
        [train_ast, train_hand_craft, train_label, test_ast, test_hand_craft, test_label, train_aug1, train_aug2,
         test_aug1, test_aug2,
         vector_len, vocabulary_size] = obj

    else:
        train_file_instances = extract_handcraft_instances(path_train_handcraft)
        test_file_instances = extract_handcraft_instances(path_test_handcraft)

        dict_token_train = parse_source(path_train_source, train_file_instances, package_heads)
        dict_token_test = parse_source(path_test_source, test_file_instances, package_heads)

        list_dict, vector_len, vocabulary_size = transform_token_to_number([dict_token_train, dict_token_test])
        dict_encoding_train = list_dict[0]
        dict_encoding_test = list_dict[1]

        train_ast, train_hand_craft, train_label = extract_data(path_train_handcraft, dict_encoding_train)
        test_ast, test_hand_craft, test_label = extract_data(path_test_handcraft, dict_encoding_test)

        train_ast, train_hand_craft, train_label = imbalance_process(train_ast, train_hand_craft, train_label)

        obj = [train_ast, train_hand_craft, train_label, test_ast, test_hand_craft, test_label, vector_len,
               vocabulary_size]
        dump_data(path_train_and_test_set, obj)

    train_ast = torch.Tensor(train_ast).to(DEVICE)
    test_ast = torch.Tensor(test_ast).to(DEVICE)
    train_aug1 = torch.Tensor(train_aug1).to(DEVICE)
    train_aug2 = torch.Tensor(train_aug2).to(DEVICE)
    test_aug1 = torch.Tensor(test_aug1).to(DEVICE)
    test_aug2 = torch.Tensor(test_aug2).to(DEVICE)

    if train_ast.size(1) > args.maxlen:
        train_ast = train_ast.narrow(1, 0, args.maxlen)
        test_ast = test_ast.narrow(1, 0, args.maxlen)
        train_aug1 = train_aug1.narrow(1, 0, args.maxlen)
        train_aug2 = train_aug2.narrow(1, 0, args.maxlen)
        test_aug1 = test_aug1.narrow(1, 0, args.maxlen)
        test_aug2 = test_aug2.narrow(1, 0, args.maxlen)

    nn_params = {
        'DICT_SIZE': vocabulary_size + 1,
        'TOKEN_SIZE': vector_len
    }

    train_dataset = Data.TensorDataset(train_ast, torch.Tensor(train_label).to(DEVICE))
    train_aug1_dataset = Data.TensorDataset(train_aug1, torch.Tensor(train_label).to(DEVICE))
    trian_aug2_dataset = Data.TensorDataset(train_aug2, torch.Tensor(train_label).to(DEVICE))
    test_dataset = Data.TensorDataset(test_ast, torch.Tensor(test_label).to(DEVICE))
    test_aug1_dataset = Data.TensorDataset(test_aug1, torch.Tensor(test_label).to(DEVICE))
    test_aug2_dataset = Data.TensorDataset(test_aug2, torch.Tensor(test_label).to(DEVICE))

    for batch_size in args.opt_batchsize:
        train_set = MyDataset(dataset1=train_aug1_dataset, dataset2=trian_aug2_dataset, dataset3=train_dataset)
        test_set = MyDataset(dataset1=test_aug1_dataset, dataset2=test_aug2_dataset, dataset3=test_dataset)
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        train_aug_loader = Data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        test_aug_loader = Data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, drop_last=True)

        iter_test_aug = iter(test_aug_loader)

        for params_i in itertools.product(args.opt_node, args.opt_epoch, args.opt_learning_rate):

            for l in range(args.loop_size):
                model = LSTM.LSTMpred()
                clsf = cls.Classifier(input_dim=128)
                target_encoder = copy.deepcopy(model)
                for p in target_encoder.parameters():
                    p.requires_grad = False

                model.to(DEVICE)
                target_encoder.to(DEVICE)
                clsf.to(DEVICE)

                temperature = 5
                momentum = args.momentum
                init_lr = nn_params['LEARNING_RATE']
                for epoch in range(nn_params['N_EPOCH']):
                    optimizer = optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-8,
                                           weight_decay=1e-4, amsgrad=False)

                    contrastive_loss_eli5 = ContrastiveLossELI5.ContrastiveLossELI5(batch_size, temperature=0.1,
                                                                                    verbose=False)
                    inter = ContrastiveLossELI5.ContrastiveLossELI5(2, temperature=5, verbose=False)
                    MMDLoss = mmd_loss()
                    BCE = nn.BCELoss()
                    target_ema_updater = EMA(0.99)
                    total_loss = 0
                    for step, (train_aug1, train_aug2, train_original) in enumerate(train_aug_loader):

                        try:
                            test_aug1, test_aug2, test_original = iter_test_aug.next()
                        except StopIteration:
                            iter_test_aug = iter(
                                Data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, drop_last=True))
                            test_aug1, test_aug2, test_original = iter_test_aug.next()

                        batch_train_aug1_features, batch_train_aug1_labels = train_aug1
                        batch_train_aug2_features, batch_train_aug2_labels = train_aug2
                        batch_train_original_features, batch_train_original_labels = train_original
                        batch_test_aug1_features, batch_test_aug1_labels = test_aug1
                        batch_test_aug2_features, batch_test_aug2_labels = test_aug2
                        batch_test_original_features, batch_test_original_labels = test_original

                        model.train()
                        clsf.train()

                        feature_train_original, projection_train_original = model(batch_train_original_features)
                        feature_train_aug1, projection_train_aug1 = model(batch_train_aug1_features)

                        feature_test_aug1, projection_test_aug1 = model(batch_test_aug1_features)
                        feature_test_aug2, projection_test_aug2 = model(batch_test_aug2_features)

                        with torch.no_grad():
                            feature_target_aug2, projection_target_aug2 = target_encoder(batch_test_aug2_features)
                            feature_target_aug1, projection_target_aug1 = target_encoder(batch_test_aug1_features)
                            feature_target_train_aug1, projection_target_train_aug1 = target_encoder(batch_train_aug1_features)
                            feature_target_train_original, projection_target_train_original = target_encoder(batch_train_original_features)

                            projection_target_aug2.detach_()
                            projection_target_aug1.detach_()
                            projection_target_train_original.detach_()
                            projection_target_train_aug1.detach_()

                        normalized_feature_test_aug1 = F.normalize(feature_test_aug1, dim=1)

                        prediction_train_original, logits_train_original = clsf(projection_train_original)
                        prediction_train_aug1, logits_train_aug1 = clsf(projection_train_aug1)
                        prediction_test_aug1, logits_test_aug1 = clsf(projection_test_aug1)
                        prediction_test_aug2, logits_test_aug2 = clsf(projection_test_aug2)

                        mmd_transfer_loss = MMDLoss(feature_target_aug2, feature_train_aug1)
                        classification_loss = BCE(logits_train_original, batch_train_original_labels)
                        byol_loss = (loss_fn(prediction_test_aug1, projection_target_aug2.detach()) + loss_fn(prediction_test_aug2,
                                                                                               projection_target_aug1.detach())) / 2
                        byol_loss = byol_loss.mean()

                        total_loss = byol_loss + mmd_transfer_loss + classification_loss
                        update_moving_average(target_ema_updater, target_encoder, model)
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        total_loss += total_loss.item()

                    average_loss = total_loss / (step + 1)

                model.eval()
                ftrain, _ = model(train_ast)
                ftest, _ = model(test_ast)
                test_label_t = copy.deepcopy(test_label)
                for i in range(test_label.shape[1]):
                    if test_label_t[i] == 0:
                        test_label_t[i] = 2
                    if test_label_t[i] == 1:
                        test_label_t[i] = 3
                f = torch.cat((ftrain, ftest), dim=0)
                l = torch.cat((torch.tensor(train_label), torch.tensor(test_label_t)), dim=0)
                
                ftrain = ftrain.data.cpu().numpy()
                ftest = ftest.data.cpu().numpy()
                ftrain = (ftrain - np.mean(ftrain, axis=0)) / np.std(ftrain, axis=0)
                ftest = (ftest - np.mean(ftest, axis=0)) / np.std(ftest, axis=0)
                processor = RandomOverSampler()
                _x, _y = processor.fit_sample(ftest, test_label.ravel())
                clsfi = LogisticRegression(max_iter=10000)
                state = np.random.get_state()
                np.random.shuffle(_x)
                np.random.set_state(state)
                np.random.shuffle(_y)
                clsfi.fit(ftrain, train_label.ravel())
                y_pred = clsfi.predict(_x)
                f1 = f1_score(y_true=_y, y_pred=y_pred)
                note = open('./res.txt', mode='a')
                note.write(train_project_name + " -> " + test_project_name + " : " + str(f1) + "\n")
                note.close()

end_time = datetime.datetime.now()
print(end_time - start_time)
