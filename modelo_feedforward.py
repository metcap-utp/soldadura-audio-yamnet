"""
FeedForward classifier para embeddings YAMNet agregados.

Este modelo agrega los embeddings YAMNet de múltiples frames
usando estadísticas (media y desviación estándar) y luego
aplica un clasificador FeedForward simple.

Adaptado para YAMNet: embeddings de 1024 dimensiones (vs 128 de VGGish).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YAMNetAggregator(nn.Module):
    """Agrega embeddings YAMNet de múltiples frames."""
    
    def __init__(self, use_std: bool = True):
        super().__init__()
        self.use_std = use_std
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, features) o (batch, features, time)
        Returns:
            Aggregated features: (batch, features) o (batch, features*2)
        """
        if x.dim() == 3:
            if x.size(2) == 1024:
                mean = x.mean(dim=1)
                if self.use_std:
                    std = x.std(dim=1, correction=0)
                    return torch.cat([mean, std], dim=1)
                return mean
            else:
                mean = x.mean(dim=2)
                if self.use_std:
                    std = x.std(dim=2, correction=0)
                    return torch.cat([mean, std], dim=1)
                return mean
        else:
            raise ValueError(f"Expected 3D input, got {x.dim()}D")


class FeedForwardClassifier(nn.Module):
    """FeedForward classifier para features agregados."""

    def __init__(
        self,
        input_size: int = 2048,  # 1024*2 con mean+std
        hidden_sizes = None,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [1024, 512, 256]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        embedding = self.feature_extractor(x)

        if return_embedding:
            return embedding

        return self.classifier(embedding)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_embedding=True)


class FeedForwardMultiTask(nn.Module):
    """FeedForward con múltiples cabezas de clasificación."""
    
    def __init__(
        self,
        input_size: int = 2048,  # 1024*2 con mean+std
        hidden_sizes = None,
        num_classes_espesor: int = 3,
        num_classes_electrodo: int = 4,
        num_classes_corriente: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [1024, 512, 256]
        
        self.aggregator = YAMNetAggregator(use_std=True)
        
        self.feature_extractor = FeedForwardClassifier(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            num_classes=3,
            dropout=dropout,
        )
        
        del self.feature_extractor.classifier
        
        embedding_dim = hidden_sizes[-1]
        self.classifier_espesor = nn.Linear(embedding_dim, num_classes_espesor)
        self.classifier_electrodo = nn.Linear(embedding_dim, num_classes_electrodo)
        self.classifier_corriente = nn.Linear(embedding_dim, num_classes_corriente)
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        Args:
            x: YAMNet embeddings (batch, time, 1024) or pre-aggregated (batch, 2048)
        Returns:
            Dict with logits or embedding
        """
        if x.dim() == 3:
            aggregated = self.aggregator(x)
        elif x.dim() == 2 and x.size(1) == 2048:
            aggregated = x
        else:
            raise ValueError(f"Expected 3D (batch, time, 1024) or 2D (batch, 2048), got {x.shape}")
        
        embedding = self.feature_extractor(aggregated, return_embedding=True)
        
        if return_embedding:
            return embedding
        
        return {
            "logits_espesor": self.classifier_espesor(embedding),
            "logits_electrodo": self.classifier_electrodo(embedding),
            "logits_corriente": self.classifier_corriente(embedding),
        }
    
    def get_embedding(self, x: torch.Tensor):
        return self.forward(x, return_embedding=True)


def test_feedforward():
    """Test the FeedForward models."""
    batch_size = 2
    time_steps = 19
    
    aggregator = YAMNetAggregator(use_std=True)
    x = torch.randn(batch_size, time_steps, 1024)
    agg = aggregator(x)
    print(f"Input shape: {x.shape}")
    print(f"Aggregated shape: {agg.shape}")
    assert agg.shape == (batch_size, 2048)
    
    classifier = FeedForwardClassifier(input_size=2048, num_classes=3)
    out = classifier(agg)
    print(f"\nClassifier output shape: {out.shape}")
    assert out.shape == (batch_size, 3)
    
    multi_task = FeedForwardMultiTask(input_size=2048)
    out = multi_task(x)
    print(f"\nMulti-task output:")
    print(f"  Espesor: {out['logits_espesor'].shape}")
    print(f"  Electrodo: {out['logits_electrodo'].shape}")
    print(f"  Corriente: {out['logits_corriente'].shape}")
    assert out["logits_espesor"].shape == (batch_size, 3)
    assert out["logits_electrodo"].shape == (batch_size, 4)
    assert out["logits_corriente"].shape == (batch_size, 2)
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_feedforward()
