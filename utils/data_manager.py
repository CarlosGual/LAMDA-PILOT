import logging
import numpy as np
from PIL import Image
from numpy.core.fromnumeric import argsort
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCIFAR224, iImageNetR,iImageNetA,CUB, objectnet, omnibenchmark, vtab


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, args):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        # Since we check if we are in a real cl scenario several times, we store it as a property
        self._real_cl = 'real_cl' in self.args.keys() and self.args['real_cl'] == True
        if self._real_cl:
            # We are not working with increments, so must specify the number of tasks
            assert self.args['nb_tasks'] is not None, "Number of tasks must be specified for real class scenario."
            self._nb_tasks = self.args['nb_tasks']
            self._used_train_examples = []
            self._nb_train_examples_per_task = len(self._train_data) // self._nb_tasks
            self._selected_train_classes = None
        else:
            assert args['init_cls'] <= len(self._class_order), "Not enough classes."
            self._increments = [args['init_cls']]
            while sum(self._increments) + args['increment'] < len(self._class_order):
                self._increments.append(args['increment'])
            offset = len(self._class_order) - sum(self._increments)
            if offset > 0:
                self._increments.append(offset)
            
    @property
    def nb_tasks(self):
        if self._real_cl:
            return self._nb_tasks
        else:
            return len(self._increments)

    @property
    def selected_train_classes(self):
        if self._real_cl:
            if self._selected_train_classes is not None:
                return self._selected_train_classes
            else:
                return ValueError("Unique classes are calculated after the first train dataset is sampled. Please, "
                                  "run get_dataset_realcl in training mode first.")
        else:
            return ValueError("Unique classes are only available in real continual learning scenario.")

    def get_task_size(self, task):
        if self._real_cl:
            # For the moment we are going to suppose an even distribution of examples per task
            return len(self._class_order) // self._nb_tasks
        else:
            return self._increments[task]

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_realcl(
        self, source, mode, selected_classes=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets

            # First, delete the examples that have already been used
            x = np.delete(x, self._used_train_examples, axis=0)
            y = np.delete(y, self._used_train_examples, axis=0)

            # Now, select the examples for the current task
            assert x.shape[0] == y.shape[0], "Data size error! Data and labels must have the same size."
            random_idxes = np.random.choice(x.shape[0], self._nb_train_examples_per_task, replace=False)

            # Update the used examples
            self._used_train_examples.extend(random_idxes)

            # Get the data
            data = x[random_idxes]
            targets = y[random_idxes]

            self._selected_train_classes = np.unique(targets)
            continual_label_transform = ReorderTargets(self._selected_train_classes)

        elif source == "test":
            x, y = self._test_data, self._test_targets

            data, targets = [], []

            for idx in selected_classes:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
                data.append(class_data)
                targets.append(class_targets)

            data, targets = np.concatenate(data), np.concatenate(targets)

            continual_label_transform = ReorderTargets(selected_classes)

        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        return DummyDataset(data, targets, trsf, self.use_path, target_transform=continual_label_transform)





    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(
            train_data, train_targets, trsf, self.use_path
        ), DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    @staticmethod
    def _select(x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    @staticmethod
    def _select_rmm(x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False, target_transform=None):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        # For compatibility with real continual learning scenario
        if self.target_transform:
            label = self.target_transform(label)

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "cifar224":
        return iCIFAR224(args)
    elif name == "imagenetr":
        return iImageNetR(args)
    elif name == "imageneta":
        return iImageNetA()
    elif name == "cub":
        return CUB()
    elif name == "objectnet":
        return objectnet()
    elif name == "omnibenchmark":
        return omnibenchmark()
    elif name == "vtab":
        return vtab()

    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ReorderTargets(object):
    """
    Converts the class-orders to 0 -- (n-1) irrespective of order passed.
    """
    def __init__(self, class_order):
        self.class_order = np.array(class_order)

    def __call__(self, target):
        return np.where(self.class_order==target)[0][0]
