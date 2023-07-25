# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
import math
from ...utils import get_or_create_path
from ...log import get_module_logger

from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


class TransformerModel(Model):
    def __init__(
        self,
        d_feat: int = 20,
        d_model: int = 64,
        batch_size: int = 256,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        n_epochs=100,
        lr=0.0005,
        metric="",
        early_stop=5,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        n_jobs=10,
        GPU=0,
        seed=None,
        **kwargs
    ):

        # set hyper-parameters.
        self.d_model = d_model
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.n_jobs = n_jobs
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.logger = get_module_logger("TransformerModel")
        self.logger.info("Naive Transformer:" "\nbatch_size : {}" "\ndevice : {}".format(self.batch_size, self.device))

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        config = TimeSeriesTransformerConfig(
        prediction_length=1,  # 预测未来时间步的数量，这里设置为1，表示预测下一日的收益率
        context_length=24,  # 上下文长度，表示用于预测的过去观测值的数量，这里设置为30
        lags_sequence=[1, 2, 3, 4, 5, 6,],  # lags序列，表示将过去的观测值作为模型的输入，这里设置为[1, 7, 14]
        num_time_features=1,  # 时间特征数量，这里设置为1，因为只有时间步长一个时间特征
        num_static_categorical_features=1,  # 静态分类特征数量，这里设置为0，因为没有静态分类特征
        cardinality=[],  # 静态分类特征的可能取值的数量，这里为空列表
        embedding_dimension=[18],  # 嵌入维度
        num_static_real_features=1,
        encoder_layers=4,  # 编码器层数，根据需要进行调整
        decoder_layers=4,  # 解码器层数，根据需要进行调整
        d_model=20,  # 模型的维度大小，根据需要进行调整
)
        self.model = TimeSeriesTransformerForPrediction(config)
        
        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):

        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, data_loader):

        self.model.train()

        for data in data_loader:
            data = data.to(self.device)
            
            feature = data[:, :, 0:-1].to(self.device)
            
            outputs = self.model(
            past_time_features = feature[:, :30, :].to(self.device),  # 过去时间特征，取前30个时间步的数据 [2, 30, 2]
            past_values=feature[:, :30, 0].to(self.device),  # 过去观测值，取前30个时间步的数据的第一个特征 [2, 30,2]
            past_observed_mask=torch.ones(256, 35).to(self.device),  # 未来观测值的观测标记，这里将所有值都设置为1 [2, 30]

            future_values=feature[:, :30, 0].to(self.device) , # 未来观测值，取从第31个时间步开始的数据的第一个特征，形状为 [156, 24]
            future_time_features = feature[:, 0:1, :].to(self.device) , # 未来时间特征，这里用简单的时间步索引作为特征，形状为 [156, 24, 20]
            future_observed_mask = torch.ones(256, 1).to(self.device)  ,# 未来观测值的观测标记，这里将所有值都设置为1，形状为 [156, 24]
        )
            
            loss = outputs.loss
            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_loader):

        self.model.eval()

        scores = []
        losses = []

        for data in data_loader:

            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            
            outputs = self.model.generate(
            past_time_features=feature[:, :30, :].to(self.device),  # 过去时间特征，取前30个时间步的数据 [2, 30, 2]
            past_values=feature[:, :30, 0].to(self.device),  # 过去观测值，取前30个时间步的数据的第一个特征 [2, 30,2]
            past_observed_mask=torch.ones(256, 35).to(self.device),  # 未来观测值的观测标记，这里将所有值都设置为1 [2, 30]

            future_time_features=feature[:, 0:1, :].to(self.device),  # 未来时间特征，这里用简单的时间步索引作为特征，形状为 [156, 24, 20]
        )
            
            pred = outputs.sequences.squeeze()
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            score = self.metric_fn(pred, label)
            scores.append(score.item())

        return np.mean(losses), np.mean(scores)

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):

        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)

        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        train_loader = DataLoader(
            dl_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs, drop_last=True
        )
        valid_loader = DataLoader(
            dl_valid, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs, drop_last=True
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            outputs = self.model.generate(
            past_time_features=feature[:, :30, :].to(self.device),  # 过去时间特征，取前30个时间步的数据 [2, 30, 2]
            past_values=feature[:, :30, 0].to(self.device),  # 过去观测值，取前30个时间步的数据的第一个特征 [2, 30,2]
            # past_observed_mask=torch.ones(256, 35).to(self.device),  # 未来观测值的观测标记，这里将所有值都设置为1 [2, 30]

            future_time_features=feature[:, 0:1, :].to(self.device),  # 未来时间特征，这里用简单的时间步索引作为特征，形状为 [156, 24, 20]
        )
            
            pred = outputs.sequences.cpu().numpy()
            pred = np.squeeze(pred)

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=dl_test.get_index())
