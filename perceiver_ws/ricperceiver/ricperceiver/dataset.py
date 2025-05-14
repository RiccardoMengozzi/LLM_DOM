import torch, os, cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as aug
import matplotlib.pyplot as plt

from ricperceiver.text_encoder import TextEncoder
from transformers import DistilBertTokenizer


def generate_gaussian(x, y, sigma=10, h=512, w=512):
    """
    Generates a 2D Gaussian point at location x,y.

    x should be in range (-1, 1)

    sigma is the standard deviation of the generated 2D Gaussian.
    """

    # Heatmap pixel per output pixel
    mu_x = int(0.5 * (x + 1.0) * w)
    mu_y = int(0.5 * (y + 1.0) * h)

    tmp_size = sigma * 3

    # Top-left
    x1, y1 = int(mu_x - tmp_size), int(mu_y - tmp_size)

    # Bottom right
    x2, y2 = int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)
    if x1 >= w or y1 >= h or x2 < 0 or y2 < 0:
        return torch.zeros(h, w)

    size = 2 * tmp_size + 1
    tx = np.arange(0, size, 1, np.float32)
    ty = tx[:, np.newaxis]
    x0 = y0 = size // 2

    # The gaussian is not normalized, we want the center value to equal 1
    g = torch.tensor(np.exp(-((tx - x0) ** 2 + (ty - y0) ** 2) / (2 * sigma**2)))

    # Determine the bounds of the source gaussian
    g_x_min, g_x_max = max(0, -x1), min(x2, w) - x1
    g_y_min, g_y_max = max(0, -y1), min(y2, h) - y1

    # Image range
    img_x_min, img_x_max = max(0, x1), min(x2, w)
    img_y_min, img_y_max = max(0, y1), min(y2, h)

    H = torch.zeros(h, w)
    H[img_y_min:img_y_max, img_x_min:img_x_max] = g[g_y_min:g_y_max, g_x_min:g_x_max]
    return H


def heatmap2argmax(heatmap):
    B, H, W = heatmap.shape
    index = heatmap.view(B, 1, -1).argmax(dim=-1)
    pts = torch.cat([index % W, index // W], dim=1)
    return pts


class Dataset(Dataset):
    def __init__(self, data_path, img_size=512, task="reg"):

        self.data_path = data_path
        self.images_folder = "images"
        self.labels_folder = "coordinates"
        self.prompts_folder = "prompts"

        self.img_files = os.listdir(os.path.join(self.data_path, self.images_folder))

        self.img_w = img_size
        self.img_h = img_size
        self.task = task

        self.transform = aug.Compose(
            [
                aug.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=[-50, 50], val_shift_limit=[-50, 50]),
            ],
            p=1,
        )

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.text_encoder = TextEncoder(model_name="distilbert-base-uncased")

    def __len__(self):
        return len(self.img_files)

    def pre_process(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = img.transpose((2, 0, 1))
        if img.max() > 1:
            img = img / 255
        return img

    def show_sample_heatmap(self, img, heatmap, prompt=None):

        points = heatmap2argmax(heatmap).squeeze().numpy()

        img = img.squeeze().numpy().transpose((1, 2, 0))
        heatmap = heatmap.squeeze().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img)
        ax[0].set_title("Image")
        ax[1].imshow(img)
        ax[1].imshow(heatmap, alpha=0.5)
        ax[1].scatter(points[0], points[1])
        ax[1].set_title("Heatmap")
        if prompt is not None:
            ax[1].set_title(f"Heatmap\n{prompt}")
        plt.tight_layout()
        plt.show()

    def show_sample_coordinates(self, img, coordinates, prompt=None):

        img = img.squeeze().numpy().transpose((1, 2, 0))

        coordinates = coordinates.squeeze().numpy()
        coordinates *= np.array(img.shape[:2])

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(img)
        ax.set_title("Image")
        if prompt is not None:
            ax.set_title(f"{prompt}")
        ax.scatter(coordinates[0], coordinates[1], marker="x")
        plt.tight_layout()
        plt.show()

    def __getitem__(self, index):
        img_name = self.img_files[index]
        img_file = os.path.join(self.data_path, self.images_folder, img_name)
        prompt_file = img_file.replace(self.images_folder, self.prompts_folder).replace(".jpg", ".txt")
        coordinates_file = img_file.replace(self.images_folder, self.labels_folder).replace(".jpg", ".txt")

        # rgb
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        orig_h, orig_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_w, self.img_h))

        # coordinates
        coordinates = np.loadtxt(coordinates_file, delimiter=",")

        # prompt
        with open(prompt_file, "r") as f:
            prompts = f.readlines()
        prompts = [prompt.strip() for prompt in prompts]

        text_emb = self.tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"]
        text_emb = self.text_encoder(text_emb)

        ##########################################
        # augmentations
        if self.transform is not None:
            augmented = self.transform(**{"image": img})
            img_aug = augmented["image"]

        img_aug = self.pre_process(img_aug)
        img_t = torch.from_numpy(img_aug).type(torch.FloatTensor)

        if self.task == "cls":

            coord_normalized = (coordinates / np.array([orig_w, orig_h])) * 2 - 1
            label = generate_gaussian(coord_normalized[0], coord_normalized[1], sigma=10, h=self.img_h, w=self.img_w)
            label = label.unsqueeze(0).type(torch.FloatTensor)

            # data.show_sample_heatmap(img_t, label, prompt=prompts[0])

        elif self.task == "reg":

            coord_normalized = coordinates / np.array([orig_w, orig_h])
            label = torch.from_numpy(coord_normalized).type(torch.FloatTensor)

            # data.show_sample_coordinates(img_t, label, prompt=prompts[0])

        return img_t, text_emb, label, index


if __name__ == "__main__":
    dp = "/home/alessio/Downloads/perceiver_data"

    data = Dataset(data_path=dp, task="cls")
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

    for i, (img, prompts, label) in enumerate(loader):
        print(i, img.shape, prompts.shape, label.shape)
