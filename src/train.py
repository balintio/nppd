import noisebase
from noisebase.lightning import MidepochCheckpoint
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import datetime
import hydra
import os

@hydra.main(version_base=None, config_path="../conf", config_name="small_4_spp")
def main(cfg):
    output_folder = os.path.join('outputs', cfg['name'])
    os.makedirs(output_folder, exist_ok=True)

    dm = hydra.utils.instantiate(cfg['training_data'])
    model = hydra.utils.instantiate(cfg['model'])

    checkpoint_time = MidepochCheckpoint(
        dirpath=os.path.join(output_folder, 'ckpt_resume'), 
        train_time_interval=datetime.timedelta(minutes=10),
        filename='last',
        enable_version_counter=False,
        save_on_train_epoch_end=True
    )

    # Should work but ModelCheckpoint has been super buggy for me
    # Somehow saves more than 10 checkpoints, and sometimes just stops saving altogether
    # checkpoint_epoch = ModelCheckpoint(
    #     dirpath=os.path.join(output_folder, 'ckpt_epoch'), 
    #     every_n_epochs=1,
    #     save_top_k=10,
    #     monitor='val_loss',
    #     mode='min',
    #     filename='{epoch:02d}-{val_loss:.6f}',
    # )

    # Save every epoch instead
    # Needs manual cleanup of old checkpoints but at least works
    checkpoint_epoch = ModelCheckpoint(
        dirpath=os.path.join(output_folder, 'ckpt_epoch'), 
        every_n_epochs=1,
        save_top_k=-1,
        filename='{epoch:02d}-{val_loss:.6f}',
    )

    logger = TensorBoardLogger(
        save_dir=output_folder, 
        name='logs',
        version='', # save to output_folder/logs directly
    )

    trainer = L.Trainer(
        max_epochs=-1, 
        precision='16-mixed', 
        callbacks=[checkpoint_time, checkpoint_epoch],
        logger=logger,
        #devices=1
    )

    trainer.fit(
        model, 
        datamodule=dm, 
        ckpt_path='last' # looks up last.ckpt from checkpoint_time callback
    )

if __name__ == '__main__':
    main()