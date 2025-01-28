import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
from utils.writer import get_writer
import os
import numpy as np


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    if 'real_cl' not in args.keys() or args['real_cl'] is False:
        init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
        logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])

        logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["prefix"],
            args["seed"],
            args["backbone_type"],
        )

    else:
        logs_name = "logs/{}/{}/{}".format(args["model_name"], args["dataset"], args["nb_tasks"])
        logfilename = "logs/{}/{}/{}/{}_{}_{}".format(
            args["model_name"],
            args["dataset"],
            args["nb_tasks"],
            args["prefix"],
            args["seed"],
            args["backbone_type"],
        )
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    if 'wandb_enabled' in args.keys() and args["wandb_enabled"] is True:
        args["run_name"] = logfilename.replace("logs/", "").replace("/", "_")
        writer = get_writer(args)
        writer.define_metric("Task")
        writer.define_metric("Task Accuracy", step_metric="Task")
        writer.define_metric("Task Forgetting", step_metric="Task")

    else:
        writer = None

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    previous_accuracy = 0.0

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        model.after_task()
        cnn_accy, nme_accy = model.eval_task()
        if 'real_cl' in args.keys() and args['real_cl'] is True:
            # Assuming data_manager.selected_train_classes is a list or array of selected class indices
            selected_classes = data_manager.selected_train_classes
            total_classes = data_manager.nb_classes

            # Create a mask with 0s in the indexes with no class
            actual_class_mask = np.zeros(total_classes, dtype=int)
            actual_class_mask[selected_classes] = 1
        else:
            # For compatibility with no real cl scenario
            actual_class_mask = np.concatenate((np.arange(data_manager.get_task_size(task)*(task+1)), [0, 0, 0, 0, 0]))

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]    
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            # logging.info("No NME accuracy.")
            # logging.info("CNN: {}".format(cnn_accy["grouped"]))
            #
            # cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            # cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            # cnn_matrix.append(cnn_values)
            #
            # cnn_curve["top1"].append(cnn_accy["top1"])
            # cnn_curve["top5"].append(cnn_accy["top5"])
            #
            # logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            # logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            actual_accuracy = cnn_accy['top1']

            # Calculate forgetting same as nadia
            if task == 0:
                old_class_mask = actual_class_mask
                previous_accuracy = actual_accuracy
                task_forgetting = 0
            else:
                masked_confusion_matrix = cnn_accy['cm'] * old_class_mask
                old_tasks_acc =  np.around((masked_confusion_matrix.diagonal().sum() / masked_confusion_matrix.sum()) * 100, decimals=2)
                old_class_mask = actual_class_mask
                task_forgetting = np.around(previous_accuracy - old_tasks_acc, decimals=2)
                previous_accuracy = actual_accuracy

            logging.info("Average Task Accuracy (CNN): {}".format(actual_accuracy))
            logging.info("Average Task Forgetting (CNN): {}\n".format(task_forgetting))

            if writer is not None:
                log_dict = {
                    "Task Accuracy": actual_accuracy,
                    "Task Forgetting": task_forgetting,
                    "Task": int(task)
                }
                writer.add_scalars(log_dict)

    if 'print_forget' in args.keys() and args['print_forget'] is True:
        if len(cnn_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(cnn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (CNN):')
            print(np_acctable)
            logging.info('Forgetting (CNN): {}'.format(forgetting))
        if len(nme_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(nme_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            print('Accuracy Matrix (NME):')
            print(np_acctable)
        logging.info('Forgetting (NME): {}'.format(forgetting))

    writer.finish() if writer is not None else None


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))