import torch
import torch.nn as nn

class WeightedBinaryCELoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(WeightedBinaryCELoss, self).__init__()
        self.alpha = alpha

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)  # 避免 log(0)
        
        loss = -self.alpha * targets * torch.log(probs) - (1 - self.alpha) * (1 - targets) * torch.log(1 - probs)
        return loss.mean()

class MultiLabelCASL(nn.Module):
    def __init__(self,
                 gamma_pos=3,
                 gamma_neg=5,
                 class_weights=None,
                 use_entropy_for_unknown=True,
                 use_tversky=False):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.class_weights = class_weights
        self.use_entropy_for_unknown = use_entropy_for_unknown
        self.use_tversky = use_tversky

    def forward(self, multi_logits, targets):
        """
        Args:
            multi_logits: Tensor of shape [B, L]
            targets: Tensor of shape [B, L] with values in {1, 0, -1}
        Returns:
            loss_casl: scalar loss value
        """
        device = multi_logits.device
        targets = targets.float()

        mask_pos = (targets == 1).float()
        mask_neg = (targets == 0).float()
        mask_unk = (targets == -1).float()

        prob = torch.clamp(torch.sigmoid(multi_logits), min=1e-4, max=1 - 1e-4)
        class_weights = self.class_weights.to(device) if self.class_weights is not None else torch.ones(prob.size(1), device=device)

        if self.use_tversky:
            alpha_t, beta_t, eps = 0.7, 0.3, 1e-7
            tp = (prob * mask_pos).sum(dim=0)
            fp = (prob * mask_neg).sum(dim=0)
            fn = ((1 - prob) * mask_pos).sum(dim=0)
            tversky = (tp + eps) / (tp + alpha_t * fp + beta_t * fn + eps)
            focal_tversky = torch.pow(1 - tversky, 2)
            loss_casl = focal_tversky.mean()
        else:
            pos_loss = -mask_pos * torch.pow(1 - prob, self.gamma_pos) * torch.log(prob)
            neg_loss = -mask_neg * torch.pow(prob, self.gamma_neg) * torch.log(1 - prob)

            if self.use_entropy_for_unknown:
                entropy = - (prob * torch.log(prob) + (1 - prob) * torch.log(1 - prob))
                weak_loss = mask_unk * entropy
            else:
                weak_loss = -mask_unk * torch.pow(prob, self.gamma_neg) * torch.log(1 - prob)

            # Apply class weights
            pos_loss = pos_loss * class_weights
            neg_loss = neg_loss * class_weights
            weak_loss = weak_loss * class_weights

            total_loss = pos_loss + neg_loss + weak_loss
            valid_mask = mask_pos + mask_neg + mask_unk

            loss_casl = total_loss.sum() / valid_mask.sum().clamp(min=1.0)

        loss_casl = torch.nan_to_num(loss_casl, nan=0.0, posinf=1e4, neginf=-1e4)
        return loss_casl
