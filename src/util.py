import torch
import torch.distributed as dist
from noisebase.torch import lrange, tensor_like

###################
# Tonemapping
###################

def normalize_radiance(luminance, return_mean = False):
    mean = torch.mean(luminance, lrange(1, luminance.dim()), keepdim=True) + 1e-8

    if return_mean:
        return luminance / mean, mean
    else:
        return luminance / mean

def clip_logp1(x):
    return torch.log(torch.maximum(x, tensor_like(x, 0)) + 1)

###################
# Distributed
###################

def dist_cat(tensor_in):
    if dist.is_available() and dist.is_initialized():
        shape = list(tensor_in.shape)
        shape[0] *= dist.get_world_size()
        tensor_out = torch.zeros(shape, dtype=tensor_in.dtype, device=tensor_in.device)
        dist.all_gather_into_tensor(tensor_out, tensor_in)
        return tensor_out
    else:
        return tensor_in

def rank_zero():
    return dist.get_rank() == 0