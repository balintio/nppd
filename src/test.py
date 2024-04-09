import noisebase
from model import Model
import hydra
import torch
import os
import glob
import re

@hydra.main(version_base=None, config_path="../conf", config_name="large_32_spp")
def main(cfg):
    output_folder = os.path.join('outputs', cfg['name'])
    ckpt_folder = os.path.join(output_folder, 'ckpt_epoch')
    ckpt_files = glob.glob(os.path.join(ckpt_folder, "*.ckpt"))

    def extract_val_loss(filename):
        name = os.path.basename(filename)
        # Matches 'val_loss=' followed digits, a decimal point, and more digits (e.g., 'val_loss=0.123')
        return float(re.search(r'val_loss=(\d+.\d+)', name).group(1))

    best_model_path = min(ckpt_files, key=lambda x: extract_val_loss(x))

    test_set = hydra.utils.instantiate(cfg['test_data'])
    model = Model.load_from_checkpoint(**cfg['model'], checkpoint_path=best_model_path)

    test_set.save_dir(output_folder)
    for sequence in test_set:
        first = True
        for i, frame in enumerate(sequence.frames):
            frame = sequence.to_torch(frame, model.device)

            if first:
                first = False
                model.temporal = model.temporal_init(frame)

            with torch.no_grad():
                output = model.test_step(frame)

            sequence.save(i, output)
        sequence.join()
        
if __name__ == '__main__':
    main()