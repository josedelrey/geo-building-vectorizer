import torch


class SegmentationMetrics:
    def __init__(self, threshold: float = 0.5, eps: float = 1e-7) -> None:
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        self._check_shapes(logits, target, valid_mask)

        valid = valid_mask.to(dtype=torch.bool)
        prediction = torch.sigmoid(logits) >= self.threshold
        target_bool = target.to(dtype=torch.bool)

        prediction = prediction[valid]
        target_bool = target_bool[valid]

        self.tp += float((prediction & target_bool).sum().item())
        self.fp += float((prediction & ~target_bool).sum().item())
        self.fn += float((~prediction & target_bool).sum().item())
        self.tn += float((~prediction & ~target_bool).sum().item())

    def compute(self) -> dict[str, float]:
        iou = self.tp / (self.tp + self.fp + self.fn + self.eps)
        dice = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn + self.eps)
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)

        return {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
        }

    @staticmethod
    def _check_shapes(
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        expected_shape = tuple(logits.shape)

        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(
                f"logits must have shape [B, 1, H, W], got {tuple(logits.shape)}"
            )
        if tuple(target.shape) != expected_shape:
            raise ValueError(
                f"target shape must match logits shape {expected_shape}, "
                f"got {tuple(target.shape)}"
            )
        if tuple(valid_mask.shape) != expected_shape:
            raise ValueError(
                f"valid_mask shape must match logits shape {expected_shape}, "
                f"got {tuple(valid_mask.shape)}"
            )
