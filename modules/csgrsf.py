"""Attempt 11: CSODCA-guided residual spatial fusion.

The module is intentionally restricted to the bottom-up P2* -> P3 node.
It retains the original PAN ``Concat(Down(P2*), P3)`` representation and
only applies a small spatial residual calibration to the P2* branch.  This
avoids the irreversible all-node weighted sums used by the earlier Lite/SADW
experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv


class CSGRSF(nn.Module):
    """Cross-Scale Guided Residual Spatial Fusion.

    Inputs are ``[low_detail, high_semantic]`` at one common spatial scale.
    The high-semantic feature remains unchanged in the output.  A one-channel
    spatial gate, guided by both inputs, calibrates the low-detail branch with
    a zero-initialized residual scale.  The output channel count is therefore
    exactly ``C_low + C_high``, matching the original ``Concat`` node.
    """

    def __init__(self, channels, out_channels):
        super().__init__()
        if len(channels) != 2:
            raise ValueError("CSGRSF expects [low_detail, high_semantic].")
        low_channels, high_channels = channels
        if out_channels != low_channels + high_channels:
            raise ValueError(
                "CSGRSF must preserve the Concat width: "
                f"got out_channels={out_channels}, expected {low_channels + high_channels}."
            )

        # Only the gate context is projected.  The original high-scale feature
        # itself is passed to the downstream C3k2 intact.
        self.high_to_low = Conv(high_channels, low_channels, k=1, s=1)
        gate_channels = 2 * low_channels
        self.gate_context = nn.Sequential(
            nn.Conv2d(gate_channels, gate_channels, 3, padding=1, groups=gate_channels, bias=False),
            nn.BatchNorm2d(gate_channels),
            nn.SiLU(),
        )
        self.gate_out = nn.Conv2d(gate_channels, 1, 1, bias=True)

        # At initialization the node is precisely the original YAML order,
        # Concat(low_detail, high_semantic); training must earn any additional
        # calibration.
        self.residual_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, x):
        low_detail, high_semantic = x
        high_context = self.high_to_low(high_semantic)
        if high_context.shape[-2:] != low_detail.shape[-2:]:
            high_context = F.interpolate(high_context, size=low_detail.shape[-2:], mode="nearest")
            high_semantic = F.interpolate(high_semantic, size=low_detail.shape[-2:], mode="nearest")

        gate_input = torch.cat((low_detail, high_context), dim=1)
        gate = torch.sigmoid(self.gate_out(self.gate_context(gate_input)))
        calibrated_low = low_detail + self.residual_scale * gate * low_detail
        return torch.cat((calibrated_low, high_semantic), dim=1)
