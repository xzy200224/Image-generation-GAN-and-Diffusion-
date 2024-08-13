from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np



def extract(v, i, shape):
    """
    从 v 中提取第 i 个数，其中 v 的形状通常是 (T, )，而 i 的形状通常是 (batch_size, )。
    等价于 [v[index] for index in i]。
    Get the i-th number in v, and the shape of v is mostly (T, ), the shape of i is mostly (batch_size, ).
    equal to [v[index] for index in i]
    """
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out


def diffusion_schedules(beta1, beta2, T):
    '''
    提前计算各个step的alpha，这里beta是线性变化
    :param beta1: beta的下限
    :param beta2: beta的下限
    :param T: 总共的step数
    '''
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = torch.linspace(beta1, beta2, T, dtype=torch.float32)
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    # 直接计算累积乘积可能会导致数值不稳定, 故实现累积乘积的对数版
    # lpha_t_bar = torch.cumprod(alpha_t, dim=0)
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()  # alpha累乘

    sqrtab = torch.sqrt(alphabar_t)  # 根号alpha累乘
    oneover_sqrta = 1 / torch.sqrt(alpha_t)  # 1 / 根号alpha

    sqrtmab = torch.sqrt(1 - alphabar_t)  # 根号下（1-alpha累乘）
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
    # 计算前一个step的alphabar_t，第一个设为1
    alphabar_t_prev = F.pad(alphabar_t[:-1], (1, 0), value=1.0)
    # 计算t时刻方差
    variance = beta_t * (1.0 - alphabar_t_prev) / (1.0 - alphabar_t)

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}, \alpha_t累积
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}, \alpha_t累积均方根
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}, (1 - \alpha_t累积)均方根
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        "alphabar_t_prev": alphabar_t_prev,  # \bar{\alpha}_{t-1}, \alpha_t累积的前一个
        "variance": variance  # \beta_t*(1-\bar{\alpha}_{t-1})/(1-\bar{\alpha_t})
    }


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model: nn.Module, beta: Tuple[float, float], T: int, loss: list[str,]=["mse"]):
        super().__init__()
        self.model = model
        self.T = T
        self.loss_fun = loss

        # register_buffer 可以提前保存alpha相关，节约时间
        for k, v in diffusion_schedules(*beta, T).items():
            self.register_buffer(k, v)  # 将变量注册为模型的缓冲区

    def diffusion_schedules(self, beta1, beta2, T):
        '''
        提前计算各个step的alpha，这里beta是线性变化
        :param beta1: beta的下限
        :param beta2: beta的下限
        :param T: 总共的step数
        '''
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

        beta_t = torch.linspace(beta1, beta2, T, dtype=torch.float32)
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        # 直接计算累积乘积可能会导致数值不稳定, 故实现累积乘积的对数版
        # lpha_t_bar = torch.cumprod(alpha_t, dim=0)
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()  # alpha累乘

        sqrtab = torch.sqrt(alphabar_t)  # 根号alpha累乘
        oneover_sqrta = 1 / torch.sqrt(alpha_t)  # 1 / 根号alpha

        sqrtmab = torch.sqrt(1 - alphabar_t)  # 根号下（1-alpha累乘）
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab
        # 计算前一个step的alphabar_t，第一个设为1
        alphabar_t_prev = F.pad(alphabar_t[:-1], (1, 0), value=1.0)
        # 计算t时刻方差
        variance = beta_t * (1.0 - alphabar_t_prev) / (1.0 - alphabar_t)

        return {
            "alpha_t": alpha_t,  # \alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alphabar_t": alphabar_t,  # \bar{\alpha_t}, \alpha_t累积
            "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}, \alpha_t累积均方根
            "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}, (1 - \alpha_t累积)均方根
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
            "alphabar_t_prev": alphabar_t_prev,  # \bar{\alpha}_{t-1}, \alpha_t累积的前一个
            "variance": variance    # \beta_t*(1-\bar{\alpha}_{t-1})/(1-\bar{\alpha_t})
        }

    def forward(self, x_0):
        # get a random training step $t \sim Uniform({1, ..., T})$
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)

        # generate $\epsilon \sim N(0, 1)$
        epsilon = torch.randn_like(x_0)

        # predict the noise added from $x_{t-1}$ to $x_t$
        x_t = (extract(self.sqrtab, t, x_0.shape) * x_0 +
               extract(self.sqrtmab, t, x_0.shape) * epsilon)
        epsilon_theta = self.model(x_t, t)

        # get the loss
        # loss = F.mse_loss(epsilon_theta, epsilon, reduction="none").sum()
        loss = self.loss_function(epsilon_theta, epsilon)
        return loss

    def loss_function(self, pred, target):
        total_loss = 0
        if 'mse' in self.loss_fun:
            total_loss += nn.functional.mse_loss(pred, target, reduction='none').sum()
        if 'mae' in self.loss_fun:
            total_loss += nn.functional.l1_loss(pred, target, reduction='none').sum()
        return total_loss


class DDPMSampler(nn.Module):
    def __init__(self, model: nn.Module, beta: Tuple[float, float], T: int):
        super().__init__()
        self.model = model
        self.T = T

        for k, v in diffusion_schedules(*beta, T).items():
            self.register_buffer(k, v)

        # # generate T steps of beta
        # self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))
        #
        # # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        # alpha_t = 1.0 - self.beta_t
        # alpha_t_bar = torch.cumprod(alpha_t, dim=0)
        # alpha_t_bar_prev = F.pad(alpha_t_bar[:-1], (1, 0), value=1.0)
        #
        # self.register_buffer("coeff_1", torch.sqrt(1.0 / alpha_t))  #根号（1/alpha）
        # self.register_buffer("coeff_2", self.coeff_1 * (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_t_bar))#根号（1/alpha）*(1.0 - alpha)/根号(1.0 - alpha_bar)
        # self.register_buffer("posterior_variance", self.beta_t * (1.0 - alpha_t_bar_prev) / (1.0 - alpha_t_bar))

    @torch.no_grad()
    def cal_mean_variance(self, x_t, t):
        """
        Calculate the mean and variance for $q(x_{t-1} | x_t, x_0)$
        """
        epsilon_theta = self.model(x_t, t)
        mean = extract(self.oneover_sqrta, t, x_t.shape) * (x_t - extract(self.mab_over_sqrtmab, t, x_t.shape) * epsilon_theta)

        # var is a constant
        var = extract(self.variance, t, x_t.shape)

        return mean, var

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step: int):
        """
        Calculate $x_{t-1}$ according to $x_t$
        """
        # batch_size = x_t.shape[0]
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        mean, var = self.cal_mean_variance(x_t, t)
        # 噪声，最后一步不加入
        z = torch.randn_like(x_t) if time_step > 0 else 0
        # x_{t-1}
        x_t_minus_one = mean + torch.sqrt(var) * z

        if torch.isnan(x_t_minus_one).int().sum() != 0:
            raise ValueError("nan in tensor!")

        return x_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, only_return_x_0: bool = True, interval: int = 1, **kwargs):
        """
        参数:
            x_t: 标准高斯噪声。形状为 (batch_size, channels, height, width) 的张量。
            only_return_x_0: 决定采样过程中是否保存图像。如果为 True，则不保存中间图片，只返回最终结果 $x_0$。
            interval: 仅在 `only_return_x_0 = False` 时有效。决定保存中间过程图片的间隔。
                无论 `interval` 的值如何，都会包含 $x_t$ 和 $x_0$。
            kwargs: 没有实际意义，仅用于兼容性。

        返回:
            如果 `only_return_x_0 = True`，则返回形状为 (batch_size, channels, height, width) 的张量，
            否则返回形状为 (batch_size, sample, channels, height, width) 的张量，包含中间图片。
        """
        x = [x_t]
        with tqdm(reversed(range(self.T)), colour="#6565b5", total=self.T) as sampling_steps:
            for time_step in sampling_steps:
                x_t = self.sample_one_step(x_t, time_step)

                if not only_return_x_0 and ((self.T - time_step) % interval == 0 or time_step == 0):
                    x.append(torch.clip(x_t, -1.0, 1.0))

                sampling_steps.set_postfix(ordered_dict={"step": time_step + 1, "sample": len(x)})

        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]
        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]


class DDIMSampler(nn.Module):
    def __init__(self, model, beta: Tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        beta_t = torch.linspace(*beta, T, dtype=torch.float32)
        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - beta_t
        self.register_buffer("alpha_t_bar", torch.cumprod(alpha_t, dim=0))

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step: int, prev_time_step: int, eta: float):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)

        # predict noise using model
        epsilon_theta_t = self.model(x_t, t)

        # calculate x_{t-1}
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_one = (
                torch.sqrt(alpha_t_prev / alpha_t) * x_t +
                (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                    (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                sigma_t * epsilon_t
        )
        return x_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, steps: int = 1, method="linear", eta=0.0,
                only_return_x_0: bool = True, interval: int = 1):
        """
        Parameters:
            x_t: Standard Gaussian noise. A tensor with shape (batch_size, channels, height, width).
            steps: Sampling steps.
            method: Sampling method, can be "linear" or "quadratic".
            eta: Coefficients of sigma parameters in the paper. The value 0 indicates DDIM, 1 indicates DDPM.
            only_return_x_0: Determines whether the image is saved during the sampling process. if True,
                intermediate pictures are not saved, and only return the final result $x_0$.
            interval: This parameter is valid only when `only_return_x_0 = False`. Decide the interval at which
                to save the intermediate process pictures, according to `step`.
                $x_t$ and $x_0$ will be included, no matter what the value of `interval` is.

        Returns:
            if `only_return_x_0 = True`, will return a tensor with shape (batch_size, channels, height, width),
            otherwise, return a tensor with shape (batch_size, sample, channels, height, width),
            include intermediate pictures.
        """
        if method == "linear":
            a = self.T // steps
            time_steps = np.asarray(list(range(0, self.T, a)))
        elif method == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).astype(np.int)
        else:
            raise NotImplementedError(f"sampling method {method} is not implemented!")

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        time_steps = time_steps + 1
        # previous sequence
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        x = [x_t]
        with tqdm(reversed(range(0, steps)), colour="#6565b5", total=steps) as sampling_steps:
            for i in sampling_steps:
                x_t = self.sample_one_step(x_t, time_steps[i], time_steps_prev[i], eta)

                if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                    x.append(torch.clip(x_t, -1.0, 1.0))

                sampling_steps.set_postfix(ordered_dict={"step": i + 1, "sample": len(x)})

        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]
        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]
