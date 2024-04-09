import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.convnext import ConvUNeXt
from networks.convunet import ConvUNet
from partitioning_pyramid import PartitioningPyramid
from loss.loss import Features, SMAPE
from util import normalize_radiance, clip_logp1, dist_cat, rank_zero
from noisebase.torch import backproject_pixel_centers

# import matplotlib
# matplotlib.use('webagg')
# import matplotlib.pyplot as plt
# import numpy as np

# def st(t):
#     return np.transpose(t.cpu().numpy(), (1, 2, 0))

from threading import Thread

class Model(L.LightningModule):
    def __init__(self, network):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(10, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(32, 32, 1)
        )
    
        self.filter = PartitioningPyramid()

        if network == '30M':
            self.weight_predictor = ConvUNeXt(
                70,
                self.filter.inputs
            )
        
        if network == '15M':
            self.weight_predictor = ConvUNet(
                70,
                self.filter.inputs
            )

        #self.features = Features(transfer='pu') # Broken with pytorch
        self.features = Features(transfer='log')
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-4)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.94)
        return [opt], [sched]
    
    def step(self, x, temporal):

        # Reprojection

        grid = backproject_pixel_centers(
            torch.mean(x['motion'], -1),
            x['crop_offset'], 
            x['prev_camera']['crop_offset'],
            as_grid=True
        )
        
        reprojected = F.grid_sample(
            temporal,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        prev_color = reprojected[:, :3]
        prev_output = reprojected[:, 3:6]
        prev_feature = reprojected[:, 6:]
    
        # Sample encoder

        batch_size = x['color'].shape[0]

        encoder_input = torch.concat((
            x['depth'],
            x['normal'],
            x['diffuse'],
            clip_logp1(normalize_radiance(x['color']))
        ), 1)

        feature = self.encoder(
            torch.permute(encoder_input, (0, 4, 1, 2, 3)).flatten(0,1)
        )
        #feature = torch.mean(feature.unflatten(0, (batch_size, -1)), 1).to(torch.float32)
        feature = torch.mean(feature.unflatten(0, (batch_size, -1)), 1)
        color = torch.mean(x['color'], -1)

        # Denoiser

        weight_predictor_input = torch.concat((
            clip_logp1(normalize_radiance(torch.concat((
                prev_color,
                color
            ), 1))),
            prev_feature,
            feature
        ), 1)
        weights = self.weight_predictor(weight_predictor_input)
        #weights = [weight.to(torch.float32) for weight in weights]

        t_lambda = torch.sigmoid(weights[0][:, self.filter.t_lambda_index, None])
        color = t_lambda * prev_color + (1 - t_lambda) * color
        feature = t_lambda * prev_feature + (1 - t_lambda) * feature

        output = self.filter(weights, color, prev_output)

        return output, torch.concat((
            color,
            output,
            feature
        ), 1), grid

    def one_step_loss(self, ref, pred):
        ref, mean = normalize_radiance(ref, True)
        pred = pred / mean

        spatial = SMAPE(ref, pred) * 0.8 * 10 + \
            self.features.spatial_loss(ref, pred) * 0.5 * 0.2
        
        return spatial, pred
    
    def two_step_loss(self, refs, preds, grid):
        refs, mean = normalize_radiance(refs, True)
        preds = preds / mean

        prev_ref = F.grid_sample(
            refs[:, 0],
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        prev_pred = F.grid_sample(
            preds[:, 0],
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        spatial = SMAPE(refs.flatten(0, 1), preds.flatten(0, 1)) * 2.0 + \
            self.features.spatial_loss(refs.flatten(0, 1), preds.flatten(0, 1)) * 0.025

        diff_ref = refs[:, 1] - prev_ref
        diff_pred = preds[:, 1] - prev_pred

        temporal = SMAPE(diff_ref, diff_pred) * 0.2 + \
            self.features.temporal_loss(refs[:, 1], preds[:, 1], prev_ref, prev_pred) * 0.025

        return spatial + temporal, preds[:, 1]
    
    def temporal_init(self, x):
        shape = list(x['reference'].shape)
        shape[1] = 38
        return torch.zeros(shape, dtype=x['reference'].dtype, device=x['reference'].device)

    def bptt_step(self, x):
        if x['frame_index'] == 0:
            temporal = self.temporal_init(x)
            y, _, _ = self.step(x, temporal)
            loss, y = self.one_step_loss(x['reference'], y)
        else:
            y_1, temporal, _    = self.step(self.prev_x, self.temporal)
            y_2, _       , grid = self.step(x, temporal)
            loss, y = self.two_step_loss(
                torch.stack((self.prev_x['reference'], x['reference']), 1),
                torch.stack((y_1, y_2), 1),
                grid
            )
        
        self.prev_x = x
        self.temporal = temporal.detach()
        return loss, y

    def training_step(self, x, batch_idx):
        loss, y = self.bptt_step(x)
            
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, x, batch_idx):
        loss, y = self.bptt_step(x)
        
        if x['frame_index'] == 63:
            y = normalize_radiance(y)
            y = torch.pow(y / (y + 1), 1/2.2)
            y = dist_cat(y).cpu().numpy()

            # Writing images can block the rank 0 process for a couple seconds
            # which often breaks distributed training so we start a separate thread
            Thread(target=self.save_images, args=(y, batch_idx, self.trainer.current_epoch)).start()
            
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def test_step(self, x):
        y, temporal, _ = self.step(x, self.temporal)
        self.temporal = temporal.detach()
        return y

    def save_images(self, images, batch_idx, epoch):
        for i, image in enumerate(images):
            self.logger.experiment.add_image(f'denoised/{batch_idx}-{i}', image, epoch)