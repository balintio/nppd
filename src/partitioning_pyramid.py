import torch
import torch.nn as nn
import torch.nn.functional as F

def splat(img, kernel, size):
    h = img.shape[2]
    w = img.shape[3]
    total = torch.zeros_like(img)

    img = F.pad(img, [(size - 1) // 2] * 4)
    kernel = F.pad(kernel, [(size - 1) // 2] * 4)

    for i in range(size):
        for j in range(size):
            total += img[:, :, i:i+h, j:j+w] * kernel[:, i*size+j, None, i:i+h, j:j+w]
    
    return total

def upscale_quadrant(img, kernel, indices):
    quad = torch.zeros(
        [img.shape[0], img.shape[1], img.shape[2] * 2, img.shape[3] * 2],
        dtype=img.dtype, device=img.device
    )
    quad[:, :, 0::2, 0::2] = img * kernel[:, indices[0], None, :, :]
    quad[:, :, 0::2, 1::2] = img * kernel[:, indices[1], None, :, :]
    quad[:, :, 1::2, 0::2] = img * kernel[:, indices[2], None, :, :]
    quad[:, :, 1::2, 1::2] = img * kernel[:, indices[3], None, :, :]
    return quad

def upscale(img, kernel):
    img = F.pad(img, (1,1,1,1))
    kernel = F.pad(kernel, (1,1,1,1))

    tl = upscale_quadrant(img, kernel, [0, 1, 4, 5])
    tr = upscale_quadrant(img, kernel, [2, 3, 6, 7])
    bl = upscale_quadrant(img, kernel, [8, 9, 12, 13])
    br = upscale_quadrant(img, kernel, [10, 11, 14,15])

    return tl[:,:,3:-1,3:-1] + tr[:,:,3:-1,1:-3] + bl[:,:,1:-3,3:-1] + br[:,:,1:-3,1:-3]

class PartitioningPyramid():
    def __init__(self, K = 5):
        self.K = K
        self.inputs = [25 + 25 + 1 + 1 + K] + [41 for i in range(K-1)]
        self.t_lambda_index = 51
    
    def __call__(self, weights, rendered, previous):   

        part_weights = F.softmax(weights[0][:, 52:], 1)
        partitions = part_weights[:, :, None] * rendered[:, None]

        denoised_levels = [
            splat(
                F.avg_pool2d(partitions[:, i], 2 ** i, 2 ** i),
                F.softmax(weights[i][:, 0:25], 1),
                5
            )
            for i in range(self.K)
        ]

        denoised = denoised_levels[-1]
        for i in reversed(range(self.K - 1)):
            denoised = denoised_levels[i] + upscale(denoised, F.softmax(weights[i+1][:, 25:41], 1) * 4)

        previous = splat(previous, F.softmax(weights[0][:, 25:50], 1), 5)
        t_mu = torch.sigmoid(weights[0][:, 50, None])

        output = t_mu * previous + (1 - t_mu) * denoised

        return output