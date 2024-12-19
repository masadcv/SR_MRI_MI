# from monai.transforms import (Compose, LoadImaged, RandFlipd, RandRotate90d,
#                               RandSpatialCropd, SpatialCropd, ToTensord)
# from torch.utils.data import DataLoader

# from dataset.transforms import RandCropScaled
# from dataset.slicedataset import SliceDataset, fetch_dir

# # define transforms for image and segmentation


# def get_transforms(splitset="train", crop_size=(224, 224)):

#     crop_size = [int(x / 4) for x in crop_size]

#     if splitset == "train":
#         return Compose(
#             [
#                 LoadImaged(keys=["lowres", "hires"], allow_missing_keys=True),
#                 # SpatialCropd(
#                 #     keys=["lowres", "hires"],
#                 #     roi_slices=[slice(4, -4), slice(4, -4)],
#                 #     allow_missing_keys=True
#                 # ),
#                 RandFlipd(
#                     keys=["lowres", "hires"],
#                     prob=0.5,
#                     allow_missing_keys=True,
#                 ),
#                 RandRotate90d(
#                     keys=["lowres", "hires"],
#                     prob=0.5,
#                     spatial_axes=(0, 1),
#                     allow_missing_keys=True,
#                 ),
#                 # RandSpatialCropd(
#                 #     keys=["lowres", "hires"], roi_size=crop_size, random_size=False
#                 # ),
#                 RandCropScaled(
#                     lowres="lowres",
#                     hires_list=["hires"],
#                     roi_size=crop_size,
#                     hires_scale=4,
#                     allow_missing_keys=True,
#                 ),
#                 ToTensord(keys=["lowres", "hires"], allow_missing_keys=True),
#             ]
#         )
#     elif splitset == "val":
#         return Compose(
#             [
#                 LoadImaged(keys=["lowres", "hires"], allow_missing_keys=True),
#                 # SpatialCropd(
#                 #     keys=["lowres", "hires"],
#                 #     roi_slices=[slice(4, -4), slice(4, -4)],
#                 #     allow_missing_keys=True
#                 # ),
#                 ToTensord(keys=["lowres", "hires"], allow_missing_keys=True),
#             ]
#         )
#     elif splitset == "test":
#         return Compose(
#             [
#                 LoadImaged(keys=["lowres", "hires"], allow_missing_keys=True),
#                 # SpatialCropd(
#                 #     keys=["lowres", "hires"],
#                 #     roi_slices=[slice(4, -4), slice(4, -4)],
#                 #     allow_missing_keys=True
#                 # ),
#                 ToTensord(keys=["lowres", "hires"], allow_missing_keys=True),
#             ]
#         )
#     else:
#         raise ValueError("Unrecognised splitset: {} received".format(splitset))

