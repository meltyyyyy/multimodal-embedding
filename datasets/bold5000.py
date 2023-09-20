import pathlib
import pickle
from collections import defaultdict

import clip as CLIP
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.dataset_util import normalize, pad_to_patch_size

_, preprocess = CLIP.load("ViT-B/32", jit=False, download_root="./.cache/clip")


# TODO : BOLD5000Base -> BOLD5000の全データをそのまま返す, BOLD5000 -> 前処理ありで返すに分ける
class BOLD5000(Dataset):
    def __init__(
        self,
        path: str,
        subject_id: str,
        num_voxels: int,
        org_num_voxels: int,
        patch_size: int,
        debug: bool = False,
        **kwargs,
    ):
        super(BOLD5000, self).__init__()
        self.data_dir = pathlib.Path(path).resolve()
        self.subject_id = subject_id
        self.data_files = []
        self.patch_size = patch_size
        self.debug = debug
        self.num_voxels = num_voxels
        self.org_num_voxels = org_num_voxels
        self.prepare_dataset()

    def prepare_dataset(self):
        # Define directories
        resp_dir = self.data_dir / "BOLD5000_GLMsingle_ROI_betas" / "py"
        stim_dir = self.data_dir / "BOLD5000_Stimuli"
        img_dir = stim_dir / "Scene_Stimuli" / "Presented_Stimuli"
        cache_dir = self.data_dir.parent.parent / ".cache" / "data" / "bold5000"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Fetch paths
        resp_files = list(resp_dir.glob(f"{self.subject_id}_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_allses_*.npy"))
        img_names_path = (
            stim_dir / "Stimuli_Presentation_Lists" / f"{self.subject_id}" / f"{self.subject_id}_imgnames.txt"
        )
        imagenet_label_path = stim_dir / "Image_Labels" / "imagenet_final_labels.txt"
        coco_annot_path = stim_dir / "Image_Labels" / "coco_final_annotations.pkl"
        coco_category_path = stim_dir / "coco_category.csv"

        assert img_names_path.exists(), f"Image names file not found: {img_names_path}"
        assert len(resp_files) > 0, f"No response files found for subject {self.subject_id}"

        # Fetch data
        img_names = [line.strip() for line in open(img_names_path, "r").readlines()]

        imagenet_labels = {}
        with open(imagenet_label_path, "r") as file:
            for line in file:
                imagenet_id, labels = line.strip().split(" ", 1)
                imagenet_labels[imagenet_id] = labels.split(", ")

        coco_labels = defaultdict(list)
        coco_category = pd.read_csv(coco_category_path).set_index("id")["name"].to_dict()
        with open(coco_annot_path, "rb") as file:
            annots = pickle.load(file)
            for image_id, annot in annots.items():
                for segment in annot:
                    if not coco_category[segment["category_id"]] in coco_labels[image_id]:
                        coco_labels[image_id].append(coco_category[segment["category_id"]])
                assert len(coco_labels[image_id]) > 0

        voxels = np.concatenate([np.load(filepath) for filepath in resp_files], axis=1)  # num_samples, num_voxels
        voxels = pad_to_patch_size(voxels, self.patch_size)
        voxels = normalize(voxels)
        voxels = np.expand_dims(voxels, axis=1)  # num_samples, 1, num_voxels_padded
        assert len(img_names) == voxels.shape[0]

        seen = set()
        for idx, (img_name, sample) in enumerate(zip(img_names, voxels)):
            if img_name in seen:
                continue

            if (img_dir / "COCO" / f"{img_name}").exists():
                img = Image.open(img_dir / "COCO" / f"{img_name}")
                # Extract 111111 from COCO_trainxxxx_000000111111.jpg
                labels = coco_labels[int(img_name.split("_")[2].split(".")[0])]
            elif (img_dir / "ImageNet" / f"{img_name}").exists():
                img = Image.open(img_dir / "ImageNet" / f"{img_name}")
                # Extract n00000000 from n00000000_xxxx.JPEG
                labels = imagenet_labels[img_name.split("_")[0]]
            elif (img_dir / "Scene" / f"{img_name}").exists():
                img = Image.open(img_dir / "Scene" / f"{img_name}")
                # Extract label from label0.jpg
                labels = [img_name.split(".")[:-1]]
            else:
                raise FileNotFoundError(f"{img_name} does not exist in the {img_dir}")

            img = preprocess(img)
            seen.add(img_name)

            cache_file = cache_dir / f"{self.subject_id}_{idx:05}.npz"
            if not cache_file.exists():
                np.savez(cache_file, resp=sample, stim=img, labels=labels)
            self.data_files.append(cache_file)

            if self.debug and len(self.data_files) > 2**10:
                return

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        data = dict(np.load(self.data_files[index]))
        resp, stim = data["resp"], data["stim"]

        if resp.shape[-1] < self.num_voxels:
            resp = np.pad(resp, ((0, 0), (0, self.num_voxels - resp.shape[-1])), "wrap")
        else:
            resp = resp[:, :, self.num_voxels]

        stim = torch.from_numpy(stim)
        resp = torch.from_numpy(resp)
        return resp, stim
