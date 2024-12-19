import csv
import os
import pathlib
import random
import xml.etree.ElementTree as etree
from typing import Sequence

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

import fastmri
from fastmri.data import transforms


def fetch_dir(key, data_config_file=pathlib.Path("fastmri_dirs.yaml")):
    """
    Data directory fetcher.

    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.

    Args:
        key (str): key to retrieve path from data_config_file.
        data_config_file (pathlib.Path,
            default=pathlib.Path("fastmri_dirs.yaml")): Default path config
            file.

    Returns:
        pathlib.Path: The path to the specified directory.
    """
    if not data_config_file.is_file():
        default_config = dict(
            knee_path="/home/chunmeifeng/Data/",
            brain_path="/home/chunmeifeng/Data/",
            # log_path="/home/chunmeifeng/experimental/MINet/",
        )
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        raise ValueError(f"Please populate {data_config_file} with directory paths.")

    with open(data_config_file, "r") as f:
        data_dir = yaml.safe_load(f)[key]

    data_dir = pathlib.Path(data_dir)

    if not data_dir.exists():
        raise ValueError(f"Path {data_dir} from {data_config_file} does not exist.")

    return data_dir


def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


class SliceDataset(Dataset):
    def __init__(
        self,
        root,
        transform,
        challenge="singlecoil",
        sample_rate=1,
        scale=2,
        mode="train",
    ):
        self.mode = mode
        self.scale = scale

        # challenge
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        # transform
        self.transform = transform

        self.examples = []

        self.cur_path = root
        self.csv_file = os.path.join(
            self.cur_path, "singlecoil_" + self.mode + "_split_less.csv"
        )

        # 读取CSV
        with open(self.csv_file, "r") as f:
            reader = csv.reader(f)

            for row in reader:
                pd_metadata, pd_num_slices = self._retrieve_metadata(
                    os.path.join(self.cur_path, row[0] + ".h5")
                )

                pdfs_metadata, pdfs_num_slices = self._retrieve_metadata(
                    os.path.join(self.cur_path, row[1] + ".h5")
                )

                for slice_id in range(min(pd_num_slices, pdfs_num_slices)):
                    self.examples.append(
                        (
                            os.path.join(self.cur_path, row[0] + ".h5"),
                            os.path.join(self.cur_path, row[1] + ".h5"),
                            slice_id,
                            pd_metadata,
                            pdfs_metadata,
                        )
                    )

        if sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)

            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):

        # 读取pd
        pd_fname, pdfs_fname, slice, pd_metadata, pdfs_metadata = self.examples[i]

        with h5py.File(pd_fname, "r") as hf:
            pd_kspace = hf["kspace"][slice]
            pd_target = hf[self.recons_key][slice] if self.recons_key in hf else None
            attrs = dict(hf.attrs)
            attrs.update(pd_metadata)

        pd_sample = self._make_sr_data(pd_kspace, pd_target, attrs, pd_fname, slice)
        
        pd_sample_dict = {"lowres": pd_sample[0].unsqueeze(0), "hires": pd_sample[1].unsqueeze(0), "attrs": pd_sample[2], "fname": pd_sample[3], "slice": pd_sample[4]}
        
        if self.transform:
            pd_sample_dict = self.transform(pd_sample_dict)

        return pd_sample_dict

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices

    def _make_sr_data(self, kspace, target, attrs, fname, slice_num):
        kspace = transforms.to_tensor(kspace)

        image = fastmri.ifft2c(kspace)

        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for sFLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = transforms.complex_center_crop(image, crop_size)

        # getLR
        imgfft = fastmri.fft2c(image)
        imgfft = transforms.complex_center_crop(
            imgfft, (320 // self.scale, 320 // self.scale)
        )
        LR_image = fastmri.ifft2c(imgfft)

        # absolute value
        LR_image = fastmri.complex_abs(LR_image)

        # normalize input
        LR_image, mean, std = transforms.normalize_instance(LR_image, eps=1e-11)
        LR_image = LR_image.clamp(-6, 6)

        # normalize target
        if target is not None:
            target = transforms.to_tensor(target)
            target = transforms.center_crop(target, crop_size)
            target = transforms.normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return LR_image, target, mean, std, fname, slice_num


def create_data_loader(
    path_to_data,
    challenge,
    subtask,
    data_transform,
    splitset,
    scale,
    batch_size=1,
    num_workers=4,
    sample_rate=1,
):
    sample_rate = sample_rate
    dataset = SliceDataset(
        root=pathlib.Path(path_to_data) / subtask / f"{challenge}_{splitset}",
        transform=data_transform,
        sample_rate=sample_rate,
        challenge=challenge,
        mode=splitset,
        scale=scale,
    )

    is_train = splitset == "train"

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=is_train,
        sampler=None,
    )

    return dataloader




if __name__ == "__main__":
    import numpy as np
    data_path = fetch_dir("knee_path")
    data_transform = None
    data_partition = "train"
    batch_size = 1
    num_workers = 4
    sample_rate = 1
    challenge = "singlecoil"
    dataloader = create_data_loader(
        data_path,
        challenge,
        data_transform,
        data_partition,
        batch_size,
        num_workers,
        sample_rate,
    )
    for i, data in enumerate(dataloader):
        print(data)
        
        lr_image = data['lowres']
        target = data['hires']

        # unnorm and same
        from PIL import Image
        lr_image = lr_image.squeeze().cpu().numpy()
        lr_image = (lr_image - lr_image.min()) / (lr_image.max() - lr_image.min())
        lr_image = Image.fromarray((lr_image * 255).astype(np.uint8))
        lr_image.save(f"lr_image_{i}.png")
        
        target = target.squeeze().cpu().numpy()
        target = (target - target.min()) / (target.max() - target.min())
        target = Image.fromarray((target * 255).astype(np.uint8))
        target.save(f"target_{i}.png")

        if i == 10:
            break