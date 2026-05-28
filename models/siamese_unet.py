import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SiameseUNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.encoder = smp.encoders.get_encoder(
            name="resnet18",
            in_channels=3,
            depth=5,
            weights="imagenet"
        )

        encoder_channels = self.encoder.out_channels

        # Decoder
        self.decoder = smp.decoders.unet.decoder.UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
        )

        # Segmentation Head
        self.segmentation_head = nn.Conv2d(
            16,
            1,
            kernel_size=1
        )

    def forward(self, pre_img, post_img):

        # Extract Features
        pre_features = self.encoder(pre_img)
        post_features = self.encoder(post_img)

        # Feature Difference
        diff_features = []

        for pre_feat, post_feat in zip(pre_features, post_features):

            diff = torch.abs(pre_feat - post_feat)

            diff_features.append(diff)

        # Decode
        decoder_output = self.decoder(diff_features)

        # Final Prediction
        mask = self.segmentation_head(decoder_output)

        return mask