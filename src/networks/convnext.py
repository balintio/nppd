import torch
import torch.nn as nn
import torch.nn.functional as F

# L119 https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py (MIT)
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# L15 https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py (MIT)
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        #x = input + self.drop_path(x)
        x = input + x
        return x

def reversedl(it):
    return list(reversed(list(it)))

class ConvUNeXt(nn.Module):
    def __init__(self,
                 in_chans,
                 out_chans,
                 depths=[4, 4, 6, 6, 8],
                 dims=[128, 128, 256, 256, 512]):
        super().__init__()
        
        self.ds_path = nn.ModuleList()
        prev_dim = in_chans

        for stage_depth, stage_dim in zip(depths, dims):
            layers = [nn.Conv2d(prev_dim, stage_dim, 1)]
            layers += [Block(stage_dim) for _ in range(stage_depth)]
            self.ds_path.append(nn.Sequential(*layers))
            prev_dim = stage_dim * 4
        
        self.us_path = nn.ModuleList()
        prev_dim = prev_dim // 4

        for stage_depth, stage_dim in reversedl(zip(depths, dims))[1:]:
            layers = [nn.Conv2d(prev_dim // 4 + stage_dim, stage_dim, 1)]
            layers += [Block(stage_dim) for _ in range(stage_depth)]
            self.us_path.append(nn.Sequential(*layers))
            prev_dim = stage_dim

        self.out_path = nn.ModuleList()
        for out_chan, stage_dim in zip(out_chans, dims):
            self.out_path.append(nn.Conv2d(stage_dim, out_chan, 1))
        
    def forward(self, x):
        skips = []

        for stage in self.ds_path:
            x = stage(x)
            skips.append(x)
            x = F.pixel_unshuffle(x, 2)
        
        x = skips[-1]
        outputs = []

        if len(self.out_path) == len(self.ds_path):
            outputs.append(self.out_path[-1](x))

        for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[1:]):
            x = torch.cat((F.pixel_shuffle(x, 2), skip), 1)
            x = stage(x)
            if i < len(self.out_path):
                outputs.append(self.out_path[i](x))
        
        return reversedl(outputs)
