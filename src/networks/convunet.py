import torch
import torch.nn as nn
import torch.nn.functional as F

def reversedl(it):
    return list(reversed(list(it)))

class ConvUNet(nn.Module):
    def __init__(self,
                 in_chans,
                 out_chans,
                 dims_and_depths=[
                    (96, 96), 
                    (96, 128), 
                    (128, 192), 
                    (192, 256), 
                    (256, 384),
                    (512, 512, 384)
                 ],
                 pool=F.max_pool2d):
        super().__init__()
        
        self.pool = pool

        self.ds_path = nn.ModuleList()
        prev_dim = in_chans

        for dims in dims_and_depths:
            layers = []

            for dim in dims:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.ds_path.append(nn.Sequential(*layers))
        
        self.us_path = nn.ModuleList()

        for dims in reversedl(dims_and_depths)[1:]:
            layers = []
            layers.append(nn.Conv2d(prev_dim + dims[-1], dims[-1], 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))
            prev_dim = dims[-1]

            for dim in reversedl(dims)[1:]:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim
            
            self.us_path.append(nn.Sequential(*layers))

        self.out_path = nn.ModuleList()
        for out_chan, dims in zip(
                out_chans, 
                dims_and_depths[:-1]+[reversedl(dims_and_depths[-1])] # Reverse bottleneck
            ):
            self.out_path.append(nn.Conv2d(dims[0], out_chan, 1))
        
    def forward(self, x):
        skips = []

        for stage in self.ds_path:
            x = stage(x)
            skips.append(x)
            x = self.pool(x, 2)
        
        x = skips[-1]
        outputs = []

        if len(self.out_path) == len(self.ds_path):
            outputs.append(self.out_path[-1](x))

        for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[1:]):
            x = torch.cat((F.interpolate(x, scale_factor=2), skip), 1)
            x = stage(x)
            if i < len(self.out_path):
                outputs.append(self.out_path[i](x))
        
        return reversedl(outputs)