"""Cross-Scale Semantic-guided ODCA used by revision 07.

CSODCA is deliberately a post-fusion module.  It receives the high-resolution
P2 BiFPN output and the adjacent P3 BiFPN feature.  P2 supplies fine location
and directional detail; aligned P3 supplies the semantic context used for
channel calibration.  The original P2 feature is retained through a residual
path, so coordinate gates never directly erase weak small-object features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNAct(nn.Sequential):
    def __init__(self, c1, c2, kernel_size=1, groups=1, activation=True):
        padding = kernel_size // 2
        layers = [nn.Conv2d(c1, c2, kernel_size, padding=padding, groups=groups, bias=False), nn.BatchNorm2d(c2)]
        if activation:
            layers.append(nn.SiLU())
        super().__init__(*layers)


class CSODCA(nn.Module):
    """Cross-Scale Semantic-guided Orthogonal Directional Channel Attention.

    Inputs are ``[F2, F3]`` where F2 is the post-fusion P2 feature and F3 is
    the adjacent P3 fusion feature.  The module implements three distinct
    operations:

    1. Coordinate-style horizontal/vertical encoding on F2;
    2. semantic channel calibration from aligned F3;
    3. residual detail enhancement from both sources.

    This is not a diagonal-attention claim.  "Orthogonal directional" means
    the two explicit horizontal and vertical coordinate encodings.
    """

    def __init__(self, channels, out_channels, direction_reduction=4, channel_reduction=4):
        super().__init__()
        if len(channels) != 2:
            raise ValueError("CSODCA expects exactly [P2_feature, P3_feature].")
        p2_channels, p3_channels = channels
        hidden_direction = max(16, out_channels // direction_reduction)
        hidden_channel = max(16, out_channels // channel_reduction)

        self.p2_projection = _ConvBNAct(p2_channels, out_channels, 1)
        self.p3_projection = _ConvBNAct(p3_channels, out_channels, 1)

        # Official coordinate-attention style shared transform, applied to the
        # semantically enriched high-resolution P2 feature.
        self.coordinate_shared = nn.Sequential(
            nn.Conv2d(out_channels, hidden_direction, 1, bias=False),
            nn.BatchNorm2d(hidden_direction),
            nn.Hardswish(),
        )
        self.horizontal_gate = nn.Conv2d(hidden_direction, out_channels, 1)
        self.vertical_gate = nn.Conv2d(hidden_direction, out_channels, 1)

        # P3 supplies category/context evidence for channel calibration.
        self.semantic_channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, hidden_channel, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden_channel, out_channels, 1, bias=True),
            nn.Sigmoid(),
        )

        # Coordinate response and semantic context are fused into a residual,
        # not multiplied onto the output as a third suppressive sigmoid gate.
        self.fusion = nn.Sequential(
            _ConvBNAct(2 * out_channels, out_channels, 1),
            _ConvBNAct(out_channels, out_channels, 3, groups=out_channels),
            _ConvBNAct(out_channels, out_channels, 1, activation=False),
        )
        # A non-zero but small start lets all branches learn immediately while
        # preserving the fused P2 feature as the dominant initial path.
        self.residual_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.capture_attention = False

    def forward(self, x):
        p2, p3 = x
        p2 = self.p2_projection(p2)
        p3 = self.p3_projection(p3)
        if p3.shape[-2:] != p2.shape[-2:]:
            p3 = F.interpolate(p3, size=p2.shape[-2:], mode="nearest")

        _, _, height, width = p2.shape
        descriptor_h = F.adaptive_avg_pool2d(p2, (height, 1))
        descriptor_w = F.adaptive_avg_pool2d(p2, (1, width)).permute(0, 1, 3, 2)
        descriptor = self.coordinate_shared(torch.cat((descriptor_h, descriptor_w), dim=2))
        descriptor_h, descriptor_w = torch.split(descriptor, (height, width), dim=2)
        descriptor_w = descriptor_w.permute(0, 1, 3, 2)
        gate_h = torch.sigmoid(self.horizontal_gate(descriptor_h))
        gate_w = torch.sigmoid(self.vertical_gate(descriptor_w))

        directional_detail = p2 * gate_h * gate_w
        semantic_gate = self.semantic_channel_gate(p3)
        semantic_context = p3 * semantic_gate
        enhancement = self.fusion(torch.cat((directional_detail, semantic_context), dim=1))
        output = p2 + self.residual_scale * enhancement

        if self.capture_attention:
            self.last_attention = {
                "horizontal_gate": gate_h.detach(),
                "vertical_gate": gate_w.detach(),
                "semantic_channel_gate": semantic_gate.detach(),
                "residual_scale": self.residual_scale.detach(),
            }
        return output
