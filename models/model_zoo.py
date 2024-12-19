# import torchvision.models # for torchvision models
from functools import partial
import models

modelname_to_func = {
    "idn": partial(
        models.IDN,
        image_features=1,
        num_features=64,
        d=16,
        s=4,
    ),
    "vdsr": models.VDSR,
}
