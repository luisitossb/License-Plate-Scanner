import torch
import torch.nn as nn
import torchvision.models as models


class LicensePlateDetector(nn.Module):
    """Pretrained ResNet backbone with a 4-output bounding-box regression head.

    Args:
        backbone: 'resnet18' (11M params, faster) or 'resnet50' (25M params, stronger).
        pretrained: load ImageNet weights when True.
    """

    def __init__(self, backbone='resnet18', pretrained=True):
        super().__init__()
        if backbone == 'resnet18':
            base     = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = 512
        elif backbone == 'resnet50':
            base     = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError(f"backbone must be 'resnet18' or 'resnet50', got '{backbone}'")

        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head     = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def compute_iou(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-sample IoU. Both tensors are (N, 4) normalized [xmin, ymin, xmax, ymax]."""
    ix1 = torch.max(preds[:, 0], targets[:, 0])
    iy1 = torch.max(preds[:, 1], targets[:, 1])
    ix2 = torch.min(preds[:, 2], targets[:, 2])
    iy2 = torch.min(preds[:, 3], targets[:, 3])

    inter  = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    area_p = (preds[:, 2]   - preds[:, 0]).clamp(0) * (preds[:, 3]   - preds[:, 1]).clamp(0)
    area_t = (targets[:, 2] - targets[:, 0]).clamp(0) * (targets[:, 3] - targets[:, 1]).clamp(0)

    return inter / (area_p + area_t - inter).clamp(min=1e-6)
