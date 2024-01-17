import torch as th

class LogNormalSampler:
    def __init__(self, p_mean=-1.2, p_std=1.2, even=False):
        self.p_mean = p_mean
        self.p_std = p_std
        self.even = even
        if self.even:
            self.inv_cdf = lambda x: th.norm.ppf(x, loc=p_mean, scale=p_std)
            self.rank, self.size = 0, 1  # 设置默认值，因为 even=False 时这些属性不会被使用

    def sample(self, bs, device):
        if self.even:
            start_i, end_i = self.rank * bs, (self.rank + 1) * bs
            global_batch_size = self.size * bs
            locs = (th.arange(start_i, end_i) + th.rand(bs)) / global_batch_size
            log_sigmas = th.tensor(self.inv_cdf(locs), dtype=th.float32, device=device)
        else:
            log_sigmas = self.p_mean + self.p_std * th.randn(bs, device=device)
        sigmas = th.exp(log_sigmas)
        weights = th.ones_like(sigmas)
        return sigmas, weights

# 创建 LogNormalSampler 实例
sampler = LogNormalSampler()

# 生成 10 个采样结果
for _ in range(1000):
    sigmas, weights = sampler.sample(1, th.device("cpu"))
    if sigmas>1:
        print("Sigma:", sigmas.item(), "Weight:", weights.item())
