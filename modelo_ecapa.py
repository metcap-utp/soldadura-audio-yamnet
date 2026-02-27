"""
Arquitectura del modelo ECAPA-TDNN para clasificación de audio SMAW.

Basado en:
- Desplanques et al. (2020) - "ECAPA-TDNN: Emphasized Channel Attention, Propagation
  and Aggregation in TDNN Based Speaker Verification"

Usa capas TDNN con atención para agregar información contextual de múltiples
escalas temporales.

Adaptado para YAMNet: embeddings de 1024 dimensiones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    """Squeeze-and-Excitation module."""

    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class ResidualBlock(nn.Module):
    """Bloque residual con SE attention."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEModule(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        return F.relu(x + residual)


class AttentiveStatisticsPooling(nn.Module):
    """Attention-based statistics pooling."""

    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 2, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.attention(torch.cat([x, x], dim=1))
        mean = (x * att).sum(dim=2) / att.sum(dim=2)
        std = torch.sqrt(((x * att) ** 2).sum(dim=2) / att.sum(dim=2) + 1e-5)
        return torch.cat([mean, std], dim=1)


class MultiHeadClassifier(nn.Module):
    """Clasificador multi-tarea para Plate, Electrode y Current."""

    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 256,
        num_classes_espesor: int = 3,
        num_classes_electrodo: int = 4,
        num_classes_corriente: int = 2,
    ):
        super().__init__()
        self.fc_emb = nn.Linear(in_dim, emb_dim)
        self.act = nn.ReLU()
        self.fc_espesor = nn.Linear(emb_dim, num_classes_espesor)
        self.fc_electrodo = nn.Linear(emb_dim, num_classes_electrodo)
        self.fc_corriente = nn.Linear(emb_dim, num_classes_corriente)

    def forward(self, stats_vec: torch.Tensor):
        emb = self.act(self.fc_emb(stats_vec))
        return {
            "embedding": emb,
            "logits_espesor": self.fc_espesor(emb),
            "logits_electrodo": self.fc_electrodo(emb),
            "logits_corriente": self.fc_corriente(emb),
        }


class SMAWECAPAModel(nn.Module):
    """Modelo completo: YAMNet -> ECAPA-TDNN -> AttentiveStatsPooling -> Classifier."""

    def __init__(
        self,
        feat_dim: int = 1024,
        ecapa_channels: int = 1024,
        emb_dim: int = 256,
        num_classes_espesor: int = 3,
        num_classes_electrodo: int = 4,
        num_classes_corriente: int = 2,
    ):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(feat_dim)
        self.layer1 = ResidualBlock(feat_dim, ecapa_channels, kernel_size=5)
        self.layer2 = ResidualBlock(ecapa_channels, ecapa_channels, kernel_size=3)
        self.layer3 = ResidualBlock(ecapa_channels, ecapa_channels, kernel_size=1)
        self.stats_pool = AttentiveStatisticsPooling(ecapa_channels)
        self.classifier = MultiHeadClassifier(
            in_dim=ecapa_channels * 2,
            emb_dim=emb_dim,
            num_classes_espesor=num_classes_espesor,
            num_classes_electrodo=num_classes_electrodo,
            num_classes_corriente=num_classes_corriente,
        )

    def forward(self, yamnet_embeddings: torch.Tensor):
        x = yamnet_embeddings
        batch_size, seq_len, feat_dim = x.shape
        x = x.transpose(1, 2)
        x = self.bn_input(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.stats_pool(x)
        return self.classifier(x)


# Alias para compatibilidad con scripts de entrenamiento
ECAPAMultiTask = SMAWECAPAModel
