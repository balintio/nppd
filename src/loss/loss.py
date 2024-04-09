import torch
import torch.nn as nn
import torch.nn.functional as F
from noisebase.torch import lrange, tensor_like
import os

def normalize_radiance(luminance, return_mean = False):
    mean = torch.mean(luminance, lrange(1, luminance.dim()), keepdim=True) + 1e-8

    if return_mean:
        return luminance / mean, mean
    else:
        return luminance / mean

def clip_logp1(x):
    return torch.log(torch.maximum(x, tensor_like(x, 0)) + 1)

def matvec(mat, vec):
    # mat, ji -> njihw
    # vec, nihw -> njihw
    # sum, i
    return torch.sum(mat[None, :, :, None, None] * vec[:, None], 2)

# Adapted from https://github.com/gfxdisp/pu21/blob/main/matlab/pu21_encoder.m
def PU21(x):
    p = [0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.09150303166, 0.9099517204, 596.3148142] #banding_glare
    rgb2xyz = tensor_like(x, [
        [0.4125, 0.3576, 0.1804],
        [0.2127, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9503]
    ])
    xyz2rgb = tensor_like(x, [
        [ 3.2407, -1.5373, -0.4986],
        [-0.9693,  1.8760,  0.0416],
        [ 0.0556, -0.2040,  1.0572]
    ])
    eps = 1e-8

    x = torch.maximum(x, tensor_like(x, 0))
    XYZ = torch.einsum('ji, nihw -> njhw', rgb2xyz, x).to(torch.float32)
    xyz = XYZ / (torch.mean(XYZ, 1, keepdim=True) + eps)

    Y = XYZ[:, 1, None]
    Y = Y * 500 + 0.005

    V = p[6] * (((p[0] + p[1] * Y**p[3])/(1 + p[2] * Y**p[3])) ** p[4] - p[5])

    XVZ = V / (xyz[:, 1, None] + eps) * xyz
    rgb = torch.einsum('ji, nihw -> njhw', xyz2rgb, XVZ).to(torch.float32) / 1000

    return rgb

class Features(nn.Module):
    def __init__(self, transfer = 'log'):
        super().__init__()

        if transfer == 'log':
            state_dict = torch.load(os.path.join(os.path.dirname(__file__), 'perceptual_segm.pt'))
            self.transfer = clip_logp1
            self.norm = 1.0
        else:
            state_dict = torch.load(os.path.join(os.path.dirname(__file__), 'perceptual_segm_pu.pt'))
            self.transfer = PU21
            self.norm = 1.6

        self.register_buffer('l0w', state_dict['model.ds_path.0.0.weight'], False)
        self.register_buffer('l0b', state_dict['model.ds_path.0.0.bias'], False)
        self.register_buffer('l1w', state_dict['model.ds_path.1.0.weight'], False)
        self.register_buffer('l1b', state_dict['model.ds_path.1.0.bias'], False)
        self.register_buffer('l2w', state_dict['model.ds_path.2.0.weight'], False)
        self.register_buffer('l2b', state_dict['model.ds_path.2.0.bias'], False)
    
    def forward(self, x):
        x = self.transfer(x)

        f0 = F.conv2d(x, self.l0w, self.l0b, padding='same')
        f1 = F.conv2d(
            F.avg_pool2d(F.leaky_relu(f0), 2),
            self.l1w, self.l1b, padding='same'
        )
        f2 = F.conv2d(
            F.avg_pool2d(F.leaky_relu(f1), 2),
            self.l2w, self.l2b, padding='same'
        )

        return f0 * self.norm, f1 * self.norm, f2 * self.norm

    def spatial_loss(self, ref, pred):
        ref_features = self(ref)
        pred_features = self(pred)

        loss = 0.0
        for ref_feature, pred_feature in zip(ref_features, pred_features):
            loss += torch.mean(torch.abs(ref_feature - pred_feature))
        return loss

    def temporal_loss(self, ref, pred, prev_ref, prev_pred):
        ref_features = self(ref)
        pred_features = self(pred)
        prev_ref_features = self(prev_ref)
        prev_pred_features = self(prev_pred)

        diff_ref = [ref_f - prev_ref_f for ref_f, prev_ref_f in zip(ref_features, prev_ref_features)]
        diff_pred = [pred_f - prev_pred_f for pred_f, prev_pred_f in zip(pred_features, prev_pred_features)]

        loss = 0.0
        for diff_ref_f, diff_pref_f in zip(diff_ref, diff_pred):
            loss += torch.mean(torch.abs(diff_ref_f - diff_pref_f))
        return loss

def SMAPE(ref, pred):
    error = torch.abs(ref - pred) / (torch.abs(ref) + torch.abs(pred) + 1e-3)
    return torch.mean(error)


if __name__ == '__main__':
    import lightning as L
    from hypersim_loader import HypersimDataModule
    from convunet import ConvUNet

    class PerceptualSegm(nn.Module):
        def __init__(self):
            super().__init__()

            self.model = ConvUNet(3, [41], [(32,)] * 5, F.avg_pool2d)
        
        def forward(self, x):
            return self.model(x)[0]

    class Model(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = PerceptualSegm()
        
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

        def preprocess(self, radiance, labels):
            radiance = clip_logp1(normalize_radiance(radiance))
            #radiance = PU21(normalize_radiance(radiance))

            labels = torch.where(labels != -1, labels, 0).to(torch.int64)
            return radiance, labels

        def training_step(self, x, batch_idx):
            x, y = self.preprocess(x['color'], x['semantic'])
            loss = F.cross_entropy(self.model(x), y)

            self.log("train_loss", loss, on_epoch=True, prog_bar=True)
            return loss
        
        def validation_step(self, x, batch_idx):
            x, y = self.preprocess(x['color'], x['semantic'])
            loss = F.cross_entropy(self.model(x), y)

            self.log("val_loss", loss, on_epoch=True)
            return loss
    
    model = Model()
    dm = HypersimDataModule()
    trainer = L.Trainer(max_epochs=500)
    trainer.fit(model, datamodule=dm)

    torch.save(model.model.state_dict(), 'perceptual_segm.pt')
    #torch.save(model.model.state_dict(), 'perceptual_segm_pu.pt')

    