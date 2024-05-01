import torch
from collections import OrderedDict

__all__ = ['load_matching_state_dict']

def load_matching_state_dict(model: torch.nn.Module, state_dict: OrderedDict[str, torch.Tensor]) -> None:
    model_state_dict: OrderedDict[str, torch.Tensor] = model.state_dict()
    matching_state_dict : OrderedDict[str, torch.Tensor]= OrderedDict()
    for key, tensor in model_state_dict.items():
        if key in state_dict and tensor.shape == state_dict[key].shape:
            matching_state_dict[key] = state_dict[key]
    model.load_state_dict(matching_state_dict, strict=False)
