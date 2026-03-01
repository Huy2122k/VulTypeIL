"""
VulTypeIL++: Scalable Replay for Vulnerability Type Identification

This package implements scalable replay strategies for continual learning
in vulnerability type identification, extending the VulTypeIL framework.
"""

__version__ = "1.0.0"
__author__ = "VulTypeIL++ Team"

from .replay_buffer import ReplayBuffer
from .mixed_dataloader import MixedBatchIterator, create_mixed_loader
from .metrics import ContinualMetrics
from .trainer import (
    OnlineEWCWithFocalLabelSmoothLoss,
    train_phase_one,
    train_phase_two,
    train_consolidation
)

__all__ = [
    'ReplayBuffer',
    'MixedBatchIterator',
    'create_mixed_loader',
    'ContinualMetrics',
    'OnlineEWCWithFocalLabelSmoothLoss',
    'train_phase_one',
    'train_phase_two',
    'train_consolidation',
]
