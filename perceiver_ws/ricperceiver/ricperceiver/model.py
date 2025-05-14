import torch
from einops import repeat

from ricperceiver.attention_rot import CrossAttentionLayer, Attention
from ricperceiver.image_encoder import Backbone


class PerceiverModelReg(torch.nn.Module):
    def __init__(
        self,
        iterations=3,
        n_latents=32,
        latent_dim=256,
        cross_heads=4,
        cross_dim_head=4,
        latent_heads=4,
        latent_dim_head=4,
        dropout=0.0,
        depth=3,
        img_encoder_type="resnet50",
    ):
        super().__init__()

        self.iterations = iterations
        self.img_encoder_type = img_encoder_type

        ############ ENCODERS ############
        # Image encoder
        self.backbone = Backbone(resnet=img_encoder_type, feature_dim=latent_dim, input_dim=3)

        # Text projector
        self.text_proj = torch.nn.Linear(768, latent_dim)

        ######## Perceiver
        self.latents = torch.nn.Parameter(torch.normal(0, 0.2, (n_latents, latent_dim)))
        self.x_projector = torch.nn.Linear(latent_dim, 1)

        self.ins_projector = torch.nn.Linear(latent_dim + latent_dim, latent_dim)

        self.cross_attention = CrossAttentionLayer(
            dim=latent_dim,
            depth=depth,
            iterations=iterations,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            dropout=dropout,
        )

        self.projector = torch.nn.Linear(latent_dim * n_latents, latent_dim)

        self.regressor_head = torch.nn.Sequential(
            torch.nn.Linear(n_latents * latent_dim, latent_dim),  ################################
            torch.nn.ReLU(),
            torch.nn.Linear(latent_dim, 2),
            torch.nn.Sigmoid(),
        )

    def forward(self, rgb, text_emb):
        """
        rgb: Bx3xHxW
        text: BxN
        """
        #############################

        # Image embedding via encoder
        img_embeddings = self.backbone(rgb)  # B x latent_dim x H/4 x W/4
        B, F, H, W = img_embeddings.shape
        ie = img_embeddings.view(B, F, -1).permute(0, 2, 1)  # Flatten the last two dimensions for each batch element
        #############################
        # Text embedding projection
        p = self.text_proj(text_emb)  # B x 8 x latent_dim
        p_for_img = p.repeat(1, H * W, 1)  # B x (H/4 * W/4) x latent_dim

        #############################
        # CAT Vision + Language
        ins = self.ins_projector(torch.cat((ie, p_for_img), dim=-1))  # B x (H/4 * W/4) x latent_dim

        #############################
        # Perceiver
        # prepare latents based on the batch size
        x = repeat(self.latents, "n d -> b n d", b=B)  # B x n_latents x latent_dim

        # Cross attention between latents query and [image+text]
        x = self.cross_attention(x, context=ins)

        # regressor head
        out = self.regressor_head(x.view(B, -1))  # B x (n_latents * latent_dim) x 2

        return out


class PerceiverModelCls(torch.nn.Module):
    def __init__(
        self,
        iterations=3,
        n_latents=32,
        latent_dim=256,
        cross_heads=4,
        cross_dim_head=4,
        latent_heads=4,
        latent_dim_head=4,
        dropout=0.0,
        depth=3,
        img_encoder_type="resnet50",
    ):
        super().__init__()

        self.iterations = iterations
        self.img_encoder_type = img_encoder_type

        ############ ENCODERS ############
        # Image encoder
        self.backbone = Backbone(resnet=img_encoder_type, feature_dim=latent_dim, input_dim=3)

        # Text projector
        self.text_proj = torch.nn.Linear(768, latent_dim)

        ######## Perceiver
        self.latents = torch.nn.Parameter(torch.normal(0, 0.2, (n_latents, latent_dim)))
        self.x_projector = torch.nn.Linear(latent_dim, 1)

        self.ins_projector = torch.nn.Linear(latent_dim + latent_dim, latent_dim)

        self.cross_attention = CrossAttentionLayer(
            dim=latent_dim,
            depth=depth,
            iterations=iterations,
            cross_heads=cross_heads,
            latent_heads=latent_heads,
            cross_dim_head=cross_dim_head,
            latent_dim_head=latent_dim_head,
            dropout=dropout,
        )

        self.projector = torch.nn.Linear(latent_dim * n_latents, latent_dim)

        # decoder cross attention
        self.decoder_cross_attn = Attention(latent_dim, heads=cross_heads, dim_head=cross_dim_head, dropout=dropout)

        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(latent_dim, latent_dim, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(latent_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(latent_dim, 1, 1),
        )

    def forward(self, rgb, text_emb):
        """
        rgb: Bx3xHxW
        text: BxN
        """
        #############################
        input_shape = rgb.shape[-2:]

        # Image embedding via encoder
        img_embeddings = self.backbone(rgb)  # B x latent_dim x H/4 x W/4
        B, F, H, W = img_embeddings.shape
        ie = img_embeddings.view(B, F, -1).permute(0, 2, 1)  # Flatten the last two dimensions for each batch element
        #############################
        # Text embedding projection
        p = self.text_proj(text_emb)  # B x 8 x latent_dim
        p_for_img = p.repeat(1, H * W, 1)  # B x (H/4 * W/4) x latent_dim

        #############################
        # CAT Vision + Language
        ins = self.ins_projector(torch.cat((ie, p_for_img), dim=-1))  # B x (H/4 * W/4) x latent_dim

        #############################
        # Perceiver
        # prepare latents based on the batch size
        x = repeat(self.latents, "n d -> b n d", b=B)  # B x n_latents x latent_dim

        # Cross attention between latents query and [image+text]
        x = self.cross_attention(x, context=ins)

        # Decoder of the latent image embedded
        x_latents = self.decoder_cross_attn(ins, context=x)

        # classification head starting from the latents
        img_latents = x_latents.permute(0, 2, 1).view(B, F, H, W)  # B x latent_dim x H x W
        out_mask = self.classifier(img_latents)
        out_mask = torch.nn.functional.interpolate(out_mask, size=input_shape, mode="bilinear", align_corners=False)

        return out_mask


if __name__ == "__main__":

    if False:
        model = PerceiverModelReg(img_encoder_type="resnet50")

        model.eval()
        x = torch.randn(1, 3, 256, 256)
        text_emb = torch.randn(1, 1, 768)
        out = model(x, text_emb)
        print(out.shape)

    if False:
        model = PerceiverModelCls(img_encoder_type="resnet50")

        model.eval()
        x = torch.randn(1, 3, 512, 512)
        text_emb = torch.randn(1, 1, 768)
        out = model(x, text_emb)
        print(out.shape)
