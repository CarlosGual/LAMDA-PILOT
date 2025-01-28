import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy


num_workers = 8
batch_size = 128

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.args = args

    def after_task(self):
        pass

    def replace_fc(self, trainloader, model, args):
        # replace fc.weight with the embedding average of train data
        model = model.eval()
        embedding_list = []
        label_list = []
        # data_list=[]
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model(data)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(trainloader.dataset.labels)
        for class_index in class_list:
            transformed_class_index = trainloader.dataset.target_transform(class_index)
            # print('Replacing...',class_index)
            data_index = (label_list == transformed_class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[transformed_class_index] = proto
        return model

    def incremental_train(self, data_manager):
        self._cur_task += 1

        logging.info(
            "Learning on task: {}".format(self._cur_task)
        )
        train_dataset = data_manager.get_dataset_realcl(
            source="train", mode="train",
        )
        selected_train_classes = data_manager.selected_train_classes
        test_dataset = data_manager.get_dataset_realcl(
            source="test", mode="test", selected_classes=selected_train_classes
        )
        # Make the protonet dataset from the two others
        train_dataset_for_protonet = copy.deepcopy(train_dataset)
        train_dataset_for_protonet.trsf = test_dataset.trsf

        self._total_classes = selected_train_classes
        self._nb_classes = data_manager.nb_classes
        self._network.update_fc(len(self._total_classes))

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers

        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers

        )

        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.args["batch_size"],
                                                    shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        
        self._network.to(self._device)
        self.replace_fc(train_loader_for_protonet, self._network, None)

        
    

   