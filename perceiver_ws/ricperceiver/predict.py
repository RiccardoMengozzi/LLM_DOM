import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn import MSELoss
plt.ion()

from ricperceiver.model import PerceiverModelReg, PerceiverModelCls
from ricperceiver.dataset import Dataset


CKPT_TO_LOAD = "/home/mengo/perceiver_ws/checkpoints/CP_cls_breezy-cloud-56_best.pth"
PATH = "/home/mengo/perceiver_ws/perceiver_data/train"
SUCCESS_TOLERANCE = 0.05
GUI = False
IMAGE_SHOW_TIME = 3

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    checkpoint = torch.load(CKPT_TO_LOAD, map_location=torch.device(device), weights_only=False)
    print(f"Checkpoint {os.path.basename(CKPT_TO_LOAD)} loaded !")



    img_size = checkpoint.get("img_size", 1024)
    encoder_type = checkpoint.get("encoder_type", "resnet50")
    feature_dim = checkpoint.get("feature_dim", 256)
    task = checkpoint.get("task", "cls")

    print(f"Image size: {img_size}")
    print(f"Encoder type: {encoder_type}")
    print(f"Feature dim: {feature_dim}")
    print(f"task: {task}")

    if task == "cls":
        model = PerceiverModelCls(img_encoder_type=encoder_type, latent_dim=feature_dim)
    elif task == "reg":
        model = PerceiverModelReg(img_encoder_type=encoder_type, latent_dim=feature_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("Model loaded !")


    dataset = Dataset(data_path=PATH, img_size=img_size, task=task)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


    criterion = MSELoss()
    total_loss = 0.0

    success_rate = 0
    with tqdm(loader) as pbar:
        for img, text_emb, label, idx in loader:
            img = img.to(device)
            text_emb = text_emb.to(device)
            label = label.to(device)

            pred = model(img, text_emb)

            if task == "reg":
                print(f"label = {label}")
                print(f"pred = {pred}")
                # calcola la MSE e accumula
                loss = criterion(pred, label)
                total_loss += loss.item()

                # (opzionale) stampa la loss per campione
                print(f"Sample {idx} – MSE Loss: {loss.item():.5f}")

            

            img = img.squeeze().detach().cpu().numpy().transpose((1, 2, 0))
            label = label.squeeze().detach().cpu().numpy()

            if task == "cls":
                pred = pred.sigmoid()

            pred = pred.squeeze().detach().cpu().numpy()


            if GUI:
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(img)
                ax[1].imshow(img)
                ax[0].set_title("Label")
                ax[1].set_title("Pred")

            if task == "reg":
                if np.linalg.norm(label - pred) < SUCCESS_TOLERANCE:
                    success_rate += 1

                label *= np.array(img.shape[:2])
                pred *= np.array(img.shape[:2])

                if GUI:
                    ax[0].scatter(label[0], label[1], marker="x", label="label")
                    ax[0].scatter(pred[0], pred[1], label="pred")

            elif task == "cls":
                label_max_idx = np.unravel_index(np.argmax(label), label.shape, order="F")
                pred_max_idx = np.unravel_index(np.argmax(pred), pred.shape, order="F")

                label_max_idx_norm = label_max_idx / np.array(label.shape)
                pred_max_idx_norm = pred_max_idx / np.array(pred.shape)


                if np.linalg.norm(np.array(label_max_idx_norm) - np.array(pred_max_idx_norm)) < SUCCESS_TOLERANCE:
                    success_rate += 1

                if GUI:
                    ax[0].scatter(label_max_idx[0], label_max_idx[1], marker="x", label="label")
                    ax[1].scatter(pred_max_idx[0], pred_max_idx[1], label="pred")
                    ax[0].imshow(label, alpha=0.25)
                    ax[1].imshow(pred, alpha=0.25)
            pbar.update(1)
            pbar.set_postfix(**{"successes": success_rate})

            if GUI:
                plt.tight_layout()
                plt.show()
                plt.show(block=False)
                plt.pause(IMAGE_SHOW_TIME)
                plt.close(fig)      # chiudi la figura
    
    success_rate = success_rate / len(loader) * 100
    print(f"Success rate: {success_rate}%")
    avg_loss = total_loss / len(loader)
    print(f"\nAverage MSE Loss on validation set: {avg_loss:.5f}")
