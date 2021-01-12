import torch as T
import dataclasses as dc
from typing import Optional, Callable


def vanilla_gradient(
        output, input,
        filter_outliers_quantiles:tuple[float,float]=[.005, .995]):
    map = T.autograd.grad(output, input)
    assert isinstance(map, tuple) and len(map) == 1, 'sanity check'
    map = map[0]
    # --> filter the bottom 0.5% and top 0.5% of gradient values since
    #     SmoothGrad paper suggests they are outliers
    low, hi = filter_outliers_quantiles
    map.clamp_(map.quantile(low), map.quantile(hi))
    return map


@dc.dataclass
class SmoothGrad:
    """Wrap a model.  Instead of outputting a prediction, generate SmoothGrad
    saliency maps for given image and an output class index to explain.

    >>> sg = SmoothGrad(model)
    >>> explanation = sg(x, index_of_class_to_explain=0)
    """
    model: T.nn.Module
    layer: Optional[T.nn.Module] = None  # defaults to a saliency map w.r.t. input
    saliency_method:str = 'vanilla_gradient'

    # smoothgrad hyperparameters
    nsamples:int = 30  # paper suggests less than 50
    std_spread:float = .15  # paper suggests values satisfying std / (max-min intensities) in [.1,.2], so std = .15*(max-min)
    apply_before_mean:Callable = lambda x: x**2  # apply a function (like absolute value or magnitude or clip extreme values) before computing mean over samples.

    def __call__(self, x: T.Tensor, index_of_class_to_explain:T.Tensor):
        explanations = []
        B,C = x.shape[:2]
        # --> compute the standard deviation per image and color channel.
        _intensity_range = x.reshape(B,C,-1).max(-1).values - x.reshape(B,C,-1).min(-1).values
        std = self.std_spread * _intensity_range
        # --> smoothgrad.  just an average of saliency maps perturbed by noise
        for i in range(self.nsamples):
            self.model.zero_grad()

            _noise = T.randn_like(x) * std.reshape(B,C,*(1 for _ in x.shape[2:]))
            x_plus_noise = (x.detach() + _noise).requires_grad_()
            yhat = self.model(x_plus_noise)

            if self.saliency_method == 'vanilla_gradient':
                map = vanilla_gradient(
                    input=self.layer if self.layer is not None else x_plus_noise,
                    output=yhat[:, index_of_class_to_explain],
                )
            else:
                raise NotImplementedError()
            map = self.apply_before_mean(map)
            explanations.append(map)
        return T.stack(explanations).mean(0)



# notes from paper
#  maybe take absolute value
#  consider .99 percentile of gradient values, because extreme values throw off input color and result in black map.

# noise, N(0, sigma^2): 10 to 20% noise?

if __name__ == "__main__":
    #  cfg = ...
    sg = SmoothGrad(cfg.model.cpu())
    x,y = cfg.train_dset[0]
    x = x.unsqueeze_(0).to(cfg.device, non_blocking=True)

    #  explanations = [sg(x, i) for i in range(y.shape[0])]
    #  e = explanations[0]
    e = sg(x,6)
