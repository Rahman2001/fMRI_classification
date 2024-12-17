import numpy as np
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from data_load_and_preprocessing.datasets import *
from utils import reproducibility

class DataHandler:
    def __init__(self, use_test=True, **config):
        self.use_test = use_test
        self.config = config
        self.dataset_name = config.get('dataset_name')
        self.split_path = Path(config.get('base_path')) / 'splits' / self.dataset_name
        self.split_path.mkdir(parents=True, exist_ok=True)
        self.split_file = self.split_path / f"split_seed_{config.get('seed')}.txt"

    def load_dataset(self):
        datasets = {'BNU_EOEC1': BNU_EOEC1}
        if self.dataset_name in datasets:
            return datasets[self.dataset_name](**self.config)
        raise NotImplementedError(f"Dataset {self.dataset_name} is not supported.")

    def split_file_exists(self):
        return self.split_file.exists()

    def create_dataloaders(self):
        reproducibility(**self.config)

        dataset = self.load_dataset()
        participant_list = np.array([str(entry[0]) for entry in dataset.index_l])

        if self.split_file_exists():
            train_names, val_names, test_names = self._read_split()
        else:
            train_names, val_names, test_names = self._random_split(participant_list)
            self._save_split(train_names, val_names, test_names)

        train_indices = self._get_indices(participant_list, train_names)
        val_indices = self._get_indices(participant_list, val_names)
        test_indices = self._get_indices(participant_list, test_names)

        train_loader = DataLoader(dataset=Subset(dataset, train_indices), **self._loader_params())
        val_loader = DataLoader(dataset=Subset(dataset, val_indices), **self._loader_params(eval_mode=True))
        test_loader = None
        if self.use_test:
            test_loader = DataLoader(dataset=Subset(dataset, test_indices), **self._loader_params(eval_mode=True))

        return train_loader, val_loader, test_loader

    def _loader_params(self, eval_mode=False):
        params = {
            'batch_size': self.config.get('batch_size'),
            'shuffle': not eval_mode,
            'num_workers': self.config.get('workers', 0) if not eval_mode else 0,
            'drop_last': not eval_mode,
            'pin_memory': self.config.get('cuda', False)
        }
        return params

    def _random_split(self, participant_list):
        train_split = self.config.get('train_split', 0.7)
        val_split = self.config.get('val_split', 0.15)

        train_names, temp_names = train_test_split(participant_list, train_size=train_split, random_state=self.config.get('seed'))
        val_names, test_names = train_test_split(temp_names, train_size=val_split / (1 - train_split), random_state=self.config.get('seed'))

        return train_names, val_names, test_names

    def _save_split(self, train_names, val_names, test_names):
        with open(self.split_file, 'w') as file:
            for name, group in [('train', train_names), ('val', val_names), ('test', test_names)]:
                file.write(f"{name}\n")
                file.writelines(f"{item}\n" for item in group)

    def _read_split(self):
        with open(self.split_file, 'r') as file:
            lines = file.read().splitlines()

        train_idx = lines.index('train') + 1
        val_idx = lines.index('val') + 1
        test_idx = lines.index('test') + 1

        train_names = lines[train_idx:val_idx - 1]
        val_names = lines[val_idx:test_idx - 1]
        test_names = lines[test_idx:]

        return train_names, val_names, test_names

    def _get_indices(self, participant_list, names):
        return np.where(np.isin(participant_list, names))[0].tolist()
