import numpy as np
import torch
from torchvision.transforms import v2

class CustomScaler(torch.nn.Module):
    def __init__(self, mode, norm_data, eps=1e-8):
        super().__init__()
        self.mode = mode
        self.eps = eps
        self.norm_data = {band: torch.tensor(norm_data.sel(norm=band).values) for band in norm_data.coords["norm"].values}

    def forward(self, tensor: torch.Tensor):
        if self.mode == "standard":
            return (tensor - self.norm_data["mean"])/(self.norm_data["std"] + self.eps)

        elif self.mode == "minmax":
            return (tensor - self.norm_data["min"])/(self.norm_data["max"] - self.norm_data["min"] + self.eps)
        
        elif self.mode =="max":
            return tensor/(self.norm_data["max"] + self.eps)

class MaskFillValue(torch.nn.Module):
    def __init__(self, mask_value=np.nan):
        super().__init__()
        self.mask_value = mask_value

    def forward(self, tensor: torch.Tensor):
        nan_mask = torch.isnan(tensor)
        if nan_mask.any():
            tensor = torch.where(nan_mask, self.mask_value, tensor)
        return tensor

class CatToOneHot(torch.nn.Module):
    def __init__(self, categories):
        super().__init__()
        self.categories = categories

    def forward(self, tensor: torch.Tensor):
        return torch.nn.functional.one_hot(tensor.squeeze(-1), num_classes=len(self.categories))
    

def get_numerical_transform(data_cube, variable, scaler_mode, mask_value=0):
    if scaler_mode is None or scaler_mode == "":
        return v2.Compose([v2.ToDtype(torch.float32)])
    else:
        return v2.Compose([v2.ToDtype(torch.float32), CustomScaler(scaler_mode, data_cube[f"{variable}_norm"]), MaskFillValue(mask_value=mask_value)])
    
def get_categorical_transform(data_cube, variable):
    categories = data_cube[variable].attrs["flag_values"]
    return v2.Compose([v2.ToDtype(torch.long), CatToOneHot(categories)])