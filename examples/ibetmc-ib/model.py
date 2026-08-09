from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def kl_divergence(alpha: torch.Tensor, classes: int) -> torch.Tensor:
    beta = torch.ones((1, classes), device=alpha.device, dtype=alpha.dtype)
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    sum_beta = beta.sum(dim=1, keepdim=True)
    log_b = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    log_b_uniform = torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(sum_beta)
    return (
        ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(
            dim=1, keepdim=True
        )
        + log_b
        + log_b_uniform
    )


def evidential_loss(
    target: torch.Tensor,
    alpha: torch.Tensor,
    classes: int,
    epoch: int,
    annealing_epoch: int,
) -> torch.Tensor:
    strength = alpha.sum(dim=1, keepdim=True)
    evidence = alpha - 1
    label = F.one_hot(target, num_classes=classes)
    fit = (label * (torch.digamma(strength) - torch.digamma(alpha))).sum(
        dim=1, keepdim=True
    )
    adjusted = evidence * (1 - label) + 1
    coefficient = min(1.0, epoch / annealing_epoch)
    return fit + coefficient * kl_divergence(adjusted, classes)


class InformationBottleneck(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(input_dim, bottleneck_dim), nn.ReLU())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(inputs)


class PseudoViewAttention(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(feature_dim, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, heads, dropout=dropout, batch_first=True
        )
        self.output = nn.Linear(embed_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, views: Sequence[torch.Tensor]) -> torch.Tensor:
        projected = [self.projection(view).unsqueeze(1) for view in views]
        values = torch.cat(projected, dim=1)
        attended, _ = self.attention(values, values, values)
        pseudo_view = self.output(attended.mean(dim=1))
        return self.dropout(self.norm(pseudo_view))


class Classifier(nn.Module):
    def __init__(self, classes: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(bottleneck_dim, classes), nn.Softplus())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class IBETMC(nn.Module):
    def __init__(
        self,
        dims: list[list[int]],
        classes: int,
        annealing_epoch: int,
        bottleneck_dim: int,
        attention_embed_dim: int,
        attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.views = len(dims)
        self.classes = classes
        self.annealing_epoch = annealing_epoch
        self.bottlenecks = nn.ModuleList(
            InformationBottleneck(view_dims[0], bottleneck_dim) for view_dims in dims
        )
        self.pseudo_view = PseudoViewAttention(
            bottleneck_dim,
            attention_embed_dim,
            attention_heads,
            dropout,
        )
        self.classifiers = nn.ModuleList(
            Classifier(classes, bottleneck_dim) for _ in range(self.views + 1)
        )

    def combine(self, alphas: list[torch.Tensor]) -> torch.Tensor:
        def combine_two(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            strengths = [value.sum(dim=1, keepdim=True) for value in (first, second)]
            evidences = [first - 1, second - 1]
            beliefs = [
                evidences[index] / strengths[index].expand_as(evidences[index])
                for index in range(2)
            ]
            uncertainties = [self.classes / strength for strength in strengths]
            conflicts = torch.bmm(
                beliefs[0].view(-1, self.classes, 1),
                beliefs[1].view(-1, 1, self.classes),
            )
            conflict = conflicts.sum(dim=(1, 2)) - torch.diagonal(
                conflicts, dim1=-2, dim2=-1
            ).sum(-1)
            denominator = (1 - conflict).view(-1, 1)
            belief = (
                beliefs[0] * beliefs[1]
                + beliefs[0] * uncertainties[1].expand_as(beliefs[0])
                + beliefs[1] * uncertainties[0].expand_as(beliefs[1])
            ) / denominator.expand_as(beliefs[0])
            uncertainty = (uncertainties[0] * uncertainties[1]) / denominator.expand_as(
                uncertainties[0]
            )
            strength = self.classes / uncertainty
            return belief * strength.expand_as(belief) + 1

        combined = alphas[0]
        for alpha in alphas[1:]:
            combined = combine_two(combined, alpha)
        return combined

    def forward(
        self,
        inputs: list[torch.Tensor],
        target: torch.Tensor,
        epoch: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        bottleneck_views = [
            bottleneck(view) for bottleneck, view in zip(self.bottlenecks, inputs)
        ]
        all_views = [*bottleneck_views, self.pseudo_view(bottleneck_views)]
        evidences = [
            classifier(view) for classifier, view in zip(self.classifiers, all_views)
        ]
        alphas = [evidence + 1 for evidence in evidences]
        combined_alpha = self.combine(alphas)
        losses = [
            evidential_loss(
                target,
                alpha,
                self.classes,
                epoch,
                self.annealing_epoch,
            )
            for alpha in [*alphas, combined_alpha]
        ]
        return evidences, combined_alpha - 1, torch.stack(losses).sum(dim=0).mean()
