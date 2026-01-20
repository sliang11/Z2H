from torch import nn


class IEDCriterion(nn.Module):

    def __init__(self, n_classes=2, **kwargs):

        super().__init__()
        self.n_classes = n_classes

    def forward(self, out_logits, target_logits):
        assert out_logits.shape == target_logits.shape
        assert out_logits.shape[-1] == (self.n_classes if self.n_classes > 2 else 1)
        return self._core(out_logits, target_logits)

    def _core(self, out_logits, target_logits):
        raise NotImplementedError()