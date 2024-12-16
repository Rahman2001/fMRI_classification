import os
import torch
from torch.utils.data import Dataset
import augmentations as aug
import pandas as pd
from pathlib import Path
import random

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    def initialize_params(self, **config):
        self.device = config.get('device', 'cpu')
        self.index_data = []
        self.global_norm = 'global_normalize'
        self.voxel_norm = 'per_voxel_normalize'
        self.random_timepoint = config.get('random_timepoint', False)
        self.setup_augmentations(**config)
        self.temporal_stride = config.get('stride_factor', 1)
        self.sequence_step = config.get('sequence_stride', 1)
        self.seq_length = config.get('sequence_length', 10)
        self.duration = self.seq_length * self.sequence_step
        self.skip_steps = list(range(0, self.duration, self.sequence_step))

    def get_input_shape(self):
        sample = torch.load(os.path.join(self.index_data[0][2], self.index_data[0][3] + '.pt'))
        return sample.squeeze().shape

    def setup_augmentations(self, **config):
        prob = config.get('augment_prob', 0)
        self.augment = aug.brain_gaussian(**config) if prob > 0 else None

    def get_timepoint_filename(self, base_name, offset):
        time_idx = [part for part in base_name.split('_') if part.isdigit()][0]
        assert len(base_name.split('_')) == 4, "Filename must follow the expected format."
        updated_name = base_name.replace(time_idx, str(int(time_idx) + offset)) + '.pt'
        return updated_name

    def resolve_timepoint(self, base_path, timepoint):
        if self.random_timepoint:
            max_offset = len(os.listdir(base_path)) - self.duration
            timepoint = f'rfMRI_r_TR_{random.randint(0, max_offset)}'
        return timepoint

    def load_sequence_data(self, base_path, timepoint):
        resolved_timepoint = self.resolve_timepoint(base_path, timepoint)
        data = torch.cat([
            torch.load(os.path.join(base_path, self.get_timepoint_filename(resolved_timepoint, step)),
                        map_location=self.device).unsqueeze(0)
            for step in self.skip_steps
        ], dim=4)

        if self.voxel_norm:
            voxel_data = torch.cat([
                torch.load(
                    os.path.join(base_path, self.get_timepoint_filename(resolved_timepoint, step)).replace(
                        self.global_norm, self.voxel_norm),
                    map_location=self.device
                ).unsqueeze(0) for step in self.skip_steps
            ], dim=4)

            voxel_data[torch.isnan(voxel_data)] = 0
            data = torch.cat([data, voxel_data], dim=0)

        return data


class BNU_EOEC1Dataset(BaseDataset):
    def __init__(self, **config):
        self.initialize_params(**config)
        self.dataset_path = Path(config.get('dataset_path', './datasets/BNU_EOEC1'))
        self.metadata_file = Path(config.get('base_path')) / 'data' / 'metadata' / 'BNU_EOEC1.csv'
        self.data_directory = self.dataset_path / 'MNI_to_TRs'
        self.meta_info = pd.read_csv(self.metadata_file)
        self.subjects = os.listdir(self.data_directory)
        self.label_mapping = {
            'open': torch.tensor([0.0]),
            'closed': torch.tensor([1.0]),
            '22-25': torch.tensor([1.0, 0.0]),
            '26-30': torch.tensor([1.0, 0.0]),
            '31-35': torch.tensor([0.0, 1.0]),
            '36+': torch.tensor([0.0, 1.0])
        }
        self._prepare_indices()

    def _prepare_indices(self):
        for idx, subject in enumerate(self.subjects):
            try:
                subject_metadata = self.meta_info[self.meta_info['SUBID_SESSION'] == int(subject)]
                age = torch.tensor(subject_metadata['AGE'].values[0])
                eye_status = subject_metadata['EYESTATUS'].values[0]
                tr_path = self.data_directory / subject / self.global_norm

                duration = len(os.listdir(tr_path))
                valid_duration = duration - self.duration
                tr_filename = os.listdir(tr_path)[0].split('_TR')[0]

                for start in range(0, valid_duration, self.temporal_stride):
                    self.index_data.append((idx, subject, str(tr_path), f"{tr_filename}_TR_{start}", valid_duration, age, eye_status))

            except KeyError as e:
                print(f"Error processing subject {subject}: {e}")

    def __len__(self):
        return len(self.index_data)

    def __getitem__(self, index):
        subj_idx, subj_id, tr_path, tr_name, duration, age, eye_status = self.index_data[index]
        age_label = self.label_mapping.get(str(age), age.float())
        sequence_data = self.load_sequence_data(tr_path, tr_name)

        if self.augment:
            sequence_data = self.augment(sequence_data)

        return {
            'fmri_sequence': sequence_data,
            'subject': subj_idx,
            'subject_binary_classification': self.label_mapping[eye_status],
            'subject_regression': age_label,
            'TR': int(tr_name.split('_')[-1])
        }
