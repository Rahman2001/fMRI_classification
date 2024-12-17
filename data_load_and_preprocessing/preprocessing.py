import os
import numpy as np
import nibabel as nib
import torch
from multiprocessing import Pool

def process_subject(subject_data):
    file_path, global_norm_path, voxel_norm_path, identifier = subject_data

    try:
        #load the fMRI data and perform trimming
        fmri_data = np.asanyarray(nib.load(file_path).dataobj)[8:-8, 8:-8, :-10, 10:]
        fmri_tensor = torch.from_numpy(fmri_data).float()

        #perform global normalization
        global_norm_tensor = normalize_global(fmri_tensor)
        save_timepoints(global_norm_tensor, global_norm_path, identifier)

        #perform per-voxel normalization
        voxel_norm_tensor = normalize_per_voxel(fmri_tensor)
        save_timepoints(voxel_norm_tensor, voxel_norm_path, identifier)

        print(f"Processed subject: {identifier}")

    except Exception as e:
        print(f"Error processing subject {identifier}: {e}")


def normalize_global(tensor):
    background_mask = tensor == 0
    foreground_mean = tensor[~background_mask].mean()
    foreground_std = tensor[~background_mask].std()
    normalized = (tensor - foreground_mean) / foreground_std
    tensor[background_mask] = normalized.min()
    tensor[~background_mask] = normalized[~background_mask]
    return tensor


def normalize_per_voxel(tensor):
    mean = tensor.mean(dim=3, keepdim=True)
    std = tensor.std(dim=3, keepdim=True)
    normalized = (tensor - mean) / std
    normalized[torch.isnan(normalized)] = 0
    return normalized


def save_timepoints(tensor, output_path, identifier):
    timepoints = torch.split(tensor, 1, dim=3)
    os.makedirs(output_path, exist_ok=True)
    for idx, timepoint in enumerate(timepoints):
        save_path = os.path.join(output_path, f"rfMRI_{identifier}_TR_{idx}.pt")
        torch.save(timepoint.clone(), save_path)


def main():
    dataset_path = r"C:\Users\Osman\Desktop\Data_Mining_2\project\datasets\Beijing_Normal_University_EOEC1"
    subjects = os.listdir(dataset_path)
    tasks = []

    for subject in subjects:
        subject_path = os.path.join(dataset_path, subject, 'rest')

        try:
            #locate the fMRI file
            fmri_file = os.path.join(subject_path, os.listdir(subject_path)[0])
            identifier = fmri_file.split('rest.nii')[0]
            print(f"Processing subject: {subject}")

            #define output directories
            global_norm_dir = os.path.join(dataset_path, 'MNI_to_TRs', subject, 'global_normalize')
            voxel_norm_dir = os.path.join(dataset_path, 'MNI_to_TRs', subject, 'per_voxel_normalize')

            tasks.append((fmri_file, global_norm_dir, voxel_norm_dir, identifier))

        except IndexError:
            print(f"Missing fMRI file for subject {subject}")

    #process subjects in parallel
    with Pool(processes=4) as pool:
        pool.map(process_subject, tasks)


if __name__ == "__main__":
    main()
