"""
Arquitectura del modelo X-Vector para clasificación de audio SMAW.

Basado en:
- Snyder et al. (2018) - "X-Vectors: Robust DNN Embeddings for Speaker Recognition"
- Desplanques et al. (2020) - "ECAPA-TDNN: Emphasized Channel Attention, Propagation
  and Aggregation in TDNN Based Speaker Verification"

La arquitectura usa BatchNorm + ReLU que estabiliza el entrenamiento y funciona
bien para clasificación supervisada multi-tarea.

Adaptado para YAMNet: embeddings de 1024 dimensiones.
"""

import torch
import torch.nn as nn


class XVector1D(nn.Module):
    """Encoder basado en X-Vector adaptado para embeddings YAMNet."""

    def __init__(self, in_channels: int = 1024, out_channels: int = 512):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 256, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class StatsPooling(nn.Module):
    """Statistics Pooling: calcula media y desviación estándar."""

    def forward(self, x):
        mean = x.mean(dim=2)
        std = x.std(dim=2, correction=0)
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


class SMAWXVectorModel(nn.Module):
    """Modelo completo: YAMNet -> XVector1D -> StatsPooling -> Classifier."""

    def __init__(
        self,
        feat_dim: int = 1024,
        xvector_dim: int = 512,
        emb_dim: int = 256,
        num_classes_espesor: int = 3,
        num_classes_electrodo: int = 4,
        num_classes_corriente: int = 2,
    ):
        super().__init__()
        self.bn_input = nn.BatchNorm1d(feat_dim, affine=False)
        self.xvector = XVector1D(in_channels=feat_dim, out_channels=xvector_dim)
        self.stats_pool = StatsPooling()
        self.classifier = MultiHeadClassifier(
            in_dim=xvector_dim * 2,
            emb_dim=emb_dim,
            num_classes_espesor=num_classes_espesor,
            num_classes_electrodo=num_classes_electrodo,
            num_classes_corriente=num_classes_corriente,
        )

    def forward(self, yamnet_embeddings: torch.Tensor):
        x = yamnet_embeddings
        batch_size, seq_len, feat_dim = x.shape
        x = x.view(batch_size * seq_len, feat_dim)
        x = self.bn_input(x)
        x = x.view(batch_size, seq_len, feat_dim)
        x = self.xvector(x)
        x = self.stats_pool(x)
        return self.classifier(x)
