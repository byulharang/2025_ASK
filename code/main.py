import os
import torch
import numpy as np
import torch.optim as optim
from net_Unet import UnetModel
from trainNet import Trainer, CheckpointManager
from dataloader import PairedDataset
from torch.utils.data import DataLoader
from lr_scheduler import get_scheduler
import warnings
import yaml
import random
from setproctitle import setproctitle



seed = 42
random.seed(seed)
np.random.seed(seed)



# config = yaml.load(open('config.yaml', "r"), Loader=yaml.FullLoader)
warnings.filterwarnings("ignore", category=FutureWarning)


# experiment setting
os.environ["CUDA_VISIBLE_DEVICES"] = "5,7,8,9"  # Do not use GPU 6
exp_order = 2
setproctitle(f"URP_inpainting_exp{exp_order}")

# file setting
log_path = f"/home/byulharang/URP/March/ablation_large/exp{exp_order}"
ckpt_path = os.path.join(log_path, "Checkpoint/")
loss_path = os.path.join(log_path, "Loss/")
train_loss_path = os.path.join(loss_path, "train_loss.csv")
val_loss_path = os.path.join(loss_path, "val_loss.csv")
test_loss_path = os.path.join(loss_path, "test_loss.csv")
metrics_path = os.path.join(loss_path, "Metrics.csv")
os.makedirs(log_path, exist_ok=True)
os.makedirs(ckpt_path, exist_ok=True)
os.makedirs(loss_path, exist_ok=True)


# Data Parallel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpus = torch.cuda.device_count()
print("Count of using GPUs:", gpus)

model = UnetModel()
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to(device)


# Dataset
Train_DT = "/home/byulharang/MP3D_ACN_N3D/Train"
Train_Depth_DT = "/home/byulharang/MP3D_ACN_N3D/Depth/Train"
Val_DT = "/home/byulharang/MP3D_ACN_N3D/Val"
Val_Depth_DT = "/home/byulharang/MP3D_ACN_N3D/Depth/Val"
Test_DT = "/home/byulharang/MP3D_ACN_N3D/Test"
Test_Depth_DT = "/home/byulharang/MP3D_ACN_N3D/Depth/Test"


# Setting
batch_size = 1 * gpus
learning_rate = 1e-4
start_epoch = 0
epochs = 200

load_exp_name = "March/ablation_large"
load_exp_order = 2
load_exp_epoch = "0001"
monitoring_exp = True if (exp_order == 0) else False

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = get_scheduler(
    optimizer,
    total_epochs=200,
    warmup_epochs=5,
    T_0=25,
    T_mult=2,
    initial_lr=learning_rate,
    eta_min=1e-6,
    start_epoch=0,
)

train_dataloader = DataLoader(
    PairedDataset(Train_DT, Train_Depth_DT),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=4 * gpus,
    pin_memory=True,
)
val_dataloader = DataLoader(
    PairedDataset(Val_DT, Val_Depth_DT),
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=gpus,
    pin_memory=True,
)
test_dataloader = DataLoader(
    PairedDataset(Test_DT, Test_Depth_DT),
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=gpus,
    pin_memory=True,
)


trainer = Trainer(
    model=model,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    test_loader=test_dataloader,
    optimizer=optimizer,
    device=device,
)
checkpoint_manager = CheckpointManager(model=model, optimizer=optimizer, device=device)


# Training setting
ckpt_load = f"/home/byulharang/URP/{load_exp_name}/exp{load_exp_order}/Checkpoint/{load_exp_epoch}.tar"
start_epoch = checkpoint_manager.load_checkpoint(ckpt_path=ckpt_load)

# Training
train_loss, val_loss, metrics = [], [], []
for epoch in range(start_epoch, epochs):
    if monitoring_exp:
        print("Monitoring Stage On")
        trainer.test(epoch, epochs, exp_order, monitor=True)
    else:
        train_loss.append(trainer.train(epoch, epochs, exp_order))
        val_loss.append(trainer.validation(epoch, epochs, exp_order))
        scheduler.step()

        # Saving checkpoint
        if (epoch + 1) % 1 == 0:
            checkpoint_manager.save_checkpoint(
                ckpt_path=f"{ckpt_path}{(epoch+1):04d}.tar", epoch=epoch
            )
            print(f"Saving checkpoint for epoch {epoch+1}")

        # Saving Train/val Loss
        np.savetxt(
            train_loss_path,
            np.column_stack((np.arange(1, len(train_loss) + 1), train_loss)),
            delimiter=",",
            header="Epoch,Train_Loss",
            comments="",
            fmt="%d,%.5f",
        )
        np.savetxt(
            val_loss_path,
            np.column_stack((np.arange(1, len(val_loss) + 1), val_loss)),
            delimiter=",",
            header="Epoch,Val_Loss",
            comments="",
            fmt="%d,%.5f",
        )
        
        print(f"saving Loss to epoch {epoch+1}")

        if (epoch +1) % 5 == 0:    # Test, Metric
            metrics.append(trainer.metrics(epoch, epochs, exp_order, metric=True))
            metrics_array = np.array(metrics)
            epochs_arr = np.arange(1, metrics_array.shape[0] + 1)
            np.savetxt(
                metrics_path,
                np.column_stack(
                    (
                        5*epochs_arr,
                        metrics_array[:, 0],
                        metrics_array[:, 1],
                        metrics_array[:, 2],
                        metrics_array[:, 3],
                    )
                ),
                delimiter=",",
                header="Epoch,PSNR,LPIPS,SSIM,Rel",
                comments="",
                fmt="%d,%.3f,%.3f,%.3f,%.3f",
            )
            print(f"saving Metric to epoch {epoch+1}")

