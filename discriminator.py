import timm
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        enable_specnorm=False,
    ):
        super().__init__()
        wrapper = spectral_norm if enable_specnorm else lambda x: x
        self.conv1 = wrapper(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(0.2)
        self.conv2 = wrapper(
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, 1, bias=False)
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(0.2)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                wrapper(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class PretrainedDiscriminator(nn.Module):
    def __init__(self, model_name="vgg16", in_channel=3, enable_specnorm=False):
        super().__init__()
        if model_name == "vgg16":
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                in_chans=in_channel,
                features_only=True,
                out_indices=(3,),
            )

            self.cls = nn.Sequential(
                ResNetBlock(512, 256, enable_specnorm=enable_specnorm),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(256, 1),
            )
        else:
            raise NotImplementedError(f"{model_name} is not implemented")

        for p in self.model.parameters():
            p.requires_grad_(False)

    def extract_features(self, x):
        x = 0.5 * (x + 1)
        with torch.no_grad():
            features = self.model(x)
        features[0].requires_grad_(True)
        return features[0]

    def get_logits(self, x):
        return self.cls(x)

    def forward(self, x):
        features = self.extract_features(x)
        return features, self.get_logits(features)

    def disable_grad(self):
        for p in self.cls.parameters():
            p.requires_grad_(False)

    def enable_grad(self):
        for p in self.cls.parameters():
            p.requires_grad_(True)


def r1_penalty(outputs, inputs, gamma=10.0):
    r1_grads = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
    )[0]
    r1_loss = r1_grads.square().sum(dim=(1, 2, 3))

    return gamma / 2 * r1_loss.mean()


class Discriminator(nn.Module):
    """
    Convolutional Discriminator
    """

    def __init__(self, in_channel=1, enable_specnorm=False):
        super(Discriminator, self).__init__()
        wrapper = nn.utils.spectral_norm if enable_specnorm else lambda x: x
        self.D = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1),  # (N, 64, 64, 64)
            ResNetBlock(64, 128, enable_specnorm=enable_specnorm),
            nn.AvgPool2d(3, 2, padding=1),  # (N, 128, 32, 32)
            ResNetBlock(128, 256, enable_specnorm=enable_specnorm),
            nn.AvgPool2d(3, 2, padding=1),  # (N, 256, 16, 16)
            ResNetBlock(256, 512, enable_specnorm=enable_specnorm),
            nn.AvgPool2d(3, 2, padding=1),  # (N, 512, 8, 8)
            ResNetBlock(512, 1024, enable_specnorm=enable_specnorm),
            nn.AvgPool2d(3, 2, padding=1),  # (N, 1024, 4, 4)
        )
        self.fc = wrapper(nn.Linear(1024 * 4 * 4, 1))  # (N, 1)

    def forward(self, x):
        B = x.size(0)
        h = self.D(x)
        h = h.view(B, -1)
        y = self.fc(h)
        return None, y

    def enable_grad(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def disable_grad(self):
        for p in self.parameters():
            p.requires_grad_(False)


if __name__ == "__main__":
    model = Discriminator()
    gamma = 0.2

    x = torch.randn(32, 3, 64, 64)
    features = model.extract_features(x)
    logits = model.get_logits(features)

    # Feature R1 regularization
    loss = r1_penalty(logits, [features], gamma=gamma)
    loss.backward()
    print("Input shape: ", x.shape)
    print("Features shape: ", features.shape)
    print("Logits shape: ", logits.shape)
    print("Loss: ", loss.item())
