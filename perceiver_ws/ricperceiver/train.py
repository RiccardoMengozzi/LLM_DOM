import os
import torch
import wandb
import numpy as np
from tqdm import tqdm
from ricperceiver.model import PerceiverModelReg, PerceiverModelCls
from ricperceiver.dataset import Dataset
from utils import set_seeds, PolyLR

import logging
from logging_formatter import ColoredFormatter

logger = logging.getLogger("Train.py")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = ColoredFormatter("[%(name)s] [%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)




default_hyperparameters = dict(
    early_stopping_patience=20,
    early_stopping_min_epochs=50,
    val_interval=1,
    warmup_steps=0,
    warmup_lr=1e-8,
    seed=0,
    ckpt_pre_train=None,
    ckpt_resume="",
    epochs=200,
    lr=1e-4,
    batchsize=4,
    encoder_type="resnet50",
    feature_dim=256,
    n_latents=32,
    dataset_path="/home/mengo/perceiver_ws/perceiver_data",
    img_size=256,
    task="cls",
    success_tolerance=0.02, # percentage of img_size
)


run = wandb.init(config=default_hyperparameters, project="ricperceiver", entity="riccardo_mengozzi", mode="online")
config = wandb.config


# set random seed
set_seeds(config.seed)


class Trainer:

    def __init__(self, config, checkpoint_dir):

        self.name = "CP_" + config["task"] + "_" + wandb.run.name
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device {self.device}")

        if config["task"] == "cls":
            self.model = PerceiverModelCls(img_encoder_type=config["encoder_type"], latent_dim=config["feature_dim"], n_latents=config["n_latents"])
        else:
            self.model = PerceiverModelReg(img_encoder_type=config["encoder_type"], latent_dim=config["feature_dim"])
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config["lr"])
        self.scheduler = PolyLR(self.optimizer, config["epochs"], power=0.97, min_lr=1e-9)


        self.criterion = torch.nn.MSELoss()


        if config["ckpt_pre_train"] is not None:
            checkpoint = torch.load(
                os.path.join("checkpoints", config["ckpt_pre_train"]), map_location=torch.device("cpu")
            )
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            current_epoch = 0
            global_step = 0
            lr = config["lr"]
            self.model.to(self.device)

        elif config["ckpt_resume"] is not None and os.path.isfile(config["ckpt_resume"]):
            checkpoint = torch.load(config["ckpt_resume"], map_location=torch.device("cpu"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            current_epoch = checkpoint["epoch"]
            global_step = checkpoint["step"]
            lr = self.optimizer.param_groups[0]["lr"]

            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            self.model.to(self.device)
            logging.info(
                f"""Resume training from: 
                current epoch:   {current_epoch}
                global step:     {global_step}
            """
            )
        else:
            logging.info("""[!] Retrain""")
            current_epoch = 0
            global_step = 0
            lr = config["lr"]
            self.model.to(self.device)

        logging.info("Starting training:")

        #############################
        self.lr = lr
        self.global_step = global_step
        self.current_epoch = current_epoch

        # Dataset ----------------------------------------------------------------------------------
        data_path_train = os.path.join(config["dataset_path"], "train")
        data_path_val = os.path.join(config["dataset_path"], "val")
        self.train_dataset = Dataset(data_path=data_path_train, img_size=config["img_size"], task=config["task"])
        self.val_dataset = Dataset(data_path=data_path_val, img_size=config["img_size"], task=config["task"])
        self.n_train = len(self.train_dataset)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config["batchsize"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config["batchsize"],
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def compute_successes(self, pred, label, task):
        if task == "cls":
            pred = pred.sigmoid()
            pred = pred.squeeze()
            label = label.squeeze()

            # if batch_size == 1, add first dimensiowith len 1
            if pred.dim() == 2:
                pred = pred.unsqueeze(0)
                label = label.unsqueeze(0)

            batch_size, height, width = pred.shape
            pred_flat = pred.view(batch_size, -1)
            label_flat = label.view(batch_size, -1)

            pred_indices = torch.argmax(pred_flat, dim=1)
            label_indices = torch.argmax(label_flat, dim=1)

            pred_coords = torch.stack([pred_indices // width, pred_indices % width], dim=1)
            label_coords = torch.stack([label_indices // width, label_indices % width], dim=1)

            pred_coords_norm = (pred_coords.float() / torch.tensor([width, height], device=self.device))
            label_coords_norm = (label_coords.float() / torch.tensor([width, height], device=self.device))

            diffs = torch.norm(pred_coords_norm.float() - label_coords_norm.float(), dim=1)
        else:
            diffs = torch.norm(pred.squeeze() - label.squeeze(), dim=1)      # shape: [batch_size]
        mean_diff = diffs.mean().detach().cpu().numpy()
        successes = (diffs < config["success_tolerance"]).sum().item() 
        return successes, mean_diff

    @torch.no_grad()
    def eval_net(self, loader):
        self.model.eval()
        val_loss = 0
        val_success_count = 0
        val_sample_count = 0
        n_val = len(loader)
        with tqdm(total=n_val, desc="Validation round", unit="batch", leave=False) as pbar:
            for img, prompt, label, idx in loader:

                img = img.to(device=self.device)
                prompt = prompt.to(device=self.device)
                label = label.to(device=self.device)

                pred = self.model(img, prompt)
                loss = self.criterion(pred, label)
                val_loss += loss.item()

                successes, mean_diff = self.compute_successes(pred, label, task=config["task"])
                val_success_count += successes
                val_sample_count  += pred.size(0)

                pbar.update()

        self.model.train()
        return val_loss / n_val, val_success_count / val_sample_count * 100, mean_diff

    def train_net(self):

        ### TRAIN
        epochs_no_improve = 0
        min_loss, min_epoch_loss = 1000, 1000
        for epoch in range(self.current_epoch, config["epochs"]):

            # TRAIN
            self.model.train()
            epoch_loss = 0
            train_success_count = 0
            train_sample_count = 0
            with tqdm(total=self.n_train, desc=f'Epoch {epoch + 1}/{config["epochs"]}', unit="img") as pbar:
                for img, prompt, label, index in self.train_loader:
                    # learning rate warmup
                    if config["warmup_steps"] > 0 and self.global_step <= config["warmup_steps"]:
                        self.lr = config["warmup_lr"] + (config["lr"] - config["warmup_lr"]) * float(
                            self.global_step / config["warmup_steps"]
                        )
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.lr

                    self.optimizer.zero_grad()

                    img = img.to(device=self.device)
                    prompt = prompt.to(device=self.device)
                    label = label.to(device=self.device)

                    pred = self.model(img, prompt)
                    loss = self.criterion(pred, label)

                    wandb.log({"train_loss": loss.item()}, step=self.global_step)

                    successes, train_mean_diff = self.compute_successes(pred, label, task=config["task"])
                    train_success_count += successes
                    train_sample_count  += pred.size(0)
                    epoch_loss += loss.item()

                    ##########
                    loss.backward()
                    self.optimizer.step()

                    # VALIDATION
                    if (self.global_step + 1) % config["val_interval"] == 0:
                        print(f"Step {self.global_step}")
                        val_loss, val_success_rate, val_mean_diff = self.eval_net(loader=self.val_loader)
                        wandb.log({"val_loss": val_loss}, step=self.global_step)
                        wandb.log({"val_success_rate": val_success_rate}, step=self.global_step)
                        wandb.log({"val_mean_diff": val_mean_diff}, step=self.global_step)
                        print(f"Evaluation loss: val {val_loss:.5f}")
                        print(f"Evaluation success rate: val {val_success_rate:.2f}%")

                    ##########
                    pbar.set_postfix(**{"loss ": loss.item(),
                                        "successes": train_success_count,
                                        "mean_diff ": train_mean_diff})

                    pbar.update(img.shape[0])
                    self.global_step += 1

            print("len(self.train_loader) = ", len(self.train_loader))
            epoch_loss /= len(self.train_loader)
            train_success_rate = train_success_count / train_sample_count * 100
            print(f"Train | loss: {epoch_loss:.5f}")
            print(f"Train | success rate: {train_success_rate:.2f}%")
            wandb.log({"train_success_rate": train_success_rate}, step=self.global_step)
            wandb.log({"train_mean_diff": train_mean_diff}, step=self.global_step)



            # SAVE LAST CHECKPOINT
            if (epoch + 1) % config["val_interval"] == 0:
                torch.save(self.get_state(), os.path.join(checkpoint_dir, self.name + "_last.pth"))
                print(f"Checkpoint saved at epoch {epoch + 1}")

            # SAVE BEST CHECKPOINT
            if val_loss < min_loss:
                min_loss = val_loss
                torch.save(self.get_state(), os.path.join(checkpoint_dir, self.name + "_best.pth"))
                print(f"*** New min validation loss {val_loss:.5f}, checkpoint BEST saved!")

            # scheduler step every epoch
            self.scheduler.step()

            # EARLY STOPPING
            if epoch_loss < min_epoch_loss:
                epochs_no_improve = 0
                min_epoch_loss = epoch_loss
            else:
                epochs_no_improve += 1

            if epoch > config["early_stopping_min_epochs"] and epochs_no_improve == config["early_stopping_patience"]:
                print("Early Stopping!")
                break

    def get_state(self):
        state = dict(config).copy()
        state["epoch"] = self.current_epoch
        state["global_step"] = self.global_step
        # state["optimizer_state_dict"] = self.optimizer.state_dict()
        state["model_state_dict"] = self.model.state_dict()
        return state


if __name__ == "__main__":
    checkpoint_dir = "checkpoints"
    trainer = Trainer(config, checkpoint_dir)
    trainer.train_net()
    run.finish()
