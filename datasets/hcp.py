import pathlib

import numpy as np
from torch.utils.data import Dataset


class HCP(Dataset):
    def __init__(self, path: str, patch_size: int, debug: bool = False, **kwargs):
        super(HCP, self).__init__()
        data_dir = pathlib.Path(path).resolve() / "npz"
        cache_dir = data_dir.parent.parent.parent / ".cache" / "data" / "hcp"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_files = []
        self.patch_size = patch_size
        self.debug = debug
        self.prepare_dataset(data_dir, cache_dir, patch_size)

    def prepare_dataset(self, data_dir: pathlib.Path, cache_dir: pathlib.Path, patch_size: int):
        for sub in data_dir.iterdir():
            if not sub.is_dir():
                continue
            for filepath in sub.glob("*"):
                if not filepath.is_file():
                    continue

                npz = dict(np.load(filepath))
                voxels = np.concatenate([npz[k] for k in npz.keys()], axis=-1)
                voxels = process_voxel_ts(voxels)
                voxels = pad_to_patch_size(voxels, patch_size)
                voxels = normalize(voxels)
                voxels = np.expand_dims(voxels, axis=1)  # num_samples, 1, num_voxels_padded

                for idx, sample in enumerate(voxels):
                    cache_file = cache_dir / f"{filepath.stem}_{sub.stem}_{idx:05}.npy"
                    if not cache_file.exists():
                        np.save(cache_file, sample)
                    self.data_files.append(cache_file)

            if self.debug and len(self.data_files) > 2**10:
                return

    @property
    def num_voxels(self):
        fmri = np.load(self.data_files[0])
        return fmri.shape[-1]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        fmri = np.load(self.data_files[index])
        return fmri


def process_voxel_ts(v, t=8):
    """
    v: voxel timeseries of a subject. (1200, num_voxels)
    t: time step of the averaging window for v. Kamitani used 8 ~ 12s
    return: voxels_reduced. reduced for the alignment of the patch size (num_samples, num_voxels_reduced)
    """
    # average the time axis first
    num_frames_per_window = t // 0.75  # ~0.75s per frame in HCP

    v_split = np.array_split(v, len(v) // num_frames_per_window, axis=0)
    v_split = np.concatenate([np.mean(f, axis=0).reshape(1, -1) for f in v_split], axis=0)
    return v_split


def pad_to_patch_size(x, patch_size):
    assert x.ndim == 2
    return np.pad(x, ((0, 0), (0, patch_size - x.shape[1] % patch_size)), "wrap")


def normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / (std * 1.0)
