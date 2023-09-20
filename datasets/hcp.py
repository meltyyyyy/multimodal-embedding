import pathlib

import numpy as np
from torch.utils.data import Dataset
from utils.dataset_util import normalize, pad_to_patch_size


# TODO : HCPBase -> HCPの全データをそのまま返す, HCP -> 前処理ありで返すに分ける
class HCP(Dataset):
    def __init__(self, path: str, patch_size: int, debug: bool = False, **kwargs):
        super(HCP, self).__init__()
        self.data_dir = pathlib.Path(path).resolve()
        self.data_files = []
        self.patch_size = patch_size
        self.debug = debug
        self.prepare_dataset(patch_size)

    def prepare_dataset(self):
        resp_dir = self.data_dir / "npz"
        cache_dir = self.data_dir.parent.parent.parent / ".cache" / "data" / "hcp"
        cache_dir.mkdir(parents=True, exist_ok=True)

        for sub in resp_dir.iterdir():
            if not sub.is_dir():
                continue
            for filepath in sub.glob("*"):
                if not filepath.is_file():
                    continue

                npz = dict(np.load(filepath))
                voxels = np.concatenate([npz[k] for k in npz.keys()], axis=-1)
                voxels = process_voxel_ts(voxels)
                voxels = pad_to_patch_size(voxels, self.patch_size)
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
        return fmri, None


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
