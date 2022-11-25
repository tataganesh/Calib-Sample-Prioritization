import torch
import torch.nn.functional as F


class LabelSmoothingLossBinary(torch.nn.Module):
    def __init__(self, smoothing=0.1, reduction="mean", weight=None):
        super(LabelSmoothingLossBinary, self).__init__()
        self.epsilon = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def linear_combination(self, x, y):
        return self.epsilon * x + (1 - self.epsilon) * y

    def forward(self, preds, target):
        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        if self.training:
            # n = preds.size(-1)
            n = 2
            bce = torch.nn.BCEWithLogitsLoss()
            bceloss = bce(preds, target)
            probs = torch.sigmoid(preds)
            log_preds = torch.log(probs)
            log_preds_inv = torch.log(1 - probs)
            loss = self.reduce_loss(-log_preds - log_preds_inv)
            return self.linear_combination(loss / n, bceloss)
        else:
            return F.binary_cross_entropy_with_logits(preds, target, weight=self.weight)


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.1, reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def linear_combination(self, x, y):
        return self.epsilon * x + (1 - self.epsilon) * y

    def forward(self, preds, target):
        if self.weight is not None:
            self.weight = self.weight.to(preds.device)

        if self.training:
            n = preds.size(-1)
            log_preds = F.log_softmax(preds, dim=-1)
            loss = self.reduce_loss(-log_preds.sum(dim=-1))
            log_preds, target, reduction = self.reduction, weight = self.weight
            nll = F.nll_loss(
                log_preds, target, reduction=self.reduction, weight=self.weight
            )
            return self.linear_combination(loss / n, nll)
        else:
            return F.binary_cross_entropy_with_logits(preds, target, weight=self.weight)
