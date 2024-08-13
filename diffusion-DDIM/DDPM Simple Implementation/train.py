import torch, time, os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from tqdm import tqdm


class ResidualConvBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

class Unet(nn.Module):
    def __init__(self, in_channels, n_feat=256):
        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),  # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t):
        '''
        输入加噪图像和对应的时间step，预测反向噪声的正态分布
        :param x: 加噪图像
        :param t: 对应step
        :return: 正态分布噪声
        '''
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # embed time step
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # 将上采样输出与step编码相加，输入到下一个上采样层
        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1 + temb1, down2)
        up3 = self.up2(up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

class DDPM(nn.Module):
    # 去噪扩散概率模型 (Denoising Diffusion Probabilistic Model, DDPM)
    def __init__(self, model, betas, n_T, device):
        """
        model: 神经网络模型（例如 U-Net），用于预测每个时间步长的噪声。
        betas: 一个包含两个元素的元组，定义了从开始到结束的 β 值范围。
        n_T: 总的时间步长数量。
        device: 模型运行所在的设备（如 'cuda' 或 'cpu'）
        """
        super(DDPM, self).__init__()
        self.model = model.to(device)
        # register_buffer 可以提前保存alpha相关，节约时间
        for k, v in self.ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)  # 将变量注册为模型的缓冲区
        self.n_T = n_T
        self.device = device
        self.loss_mse = nn.MSELoss()

    def ddpm_schedules(self, beta1, beta2, T):
        '''
        提前计算各个step的alpha，这里beta是线性变化
        :param beta1: beta的下限
        :param beta2: beta的下限
        :param T: 总共的step数
        '''
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

        beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1  # 生成beta1-beta2均匀分布的数组
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        # 直接计算累积乘积可能会导致数值不稳定，尤其是在乘积中有许多小于 1 的小数时。
        # 故改为log累加，然后exp
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()  # alpha累乘

        sqrtab = torch.sqrt(alphabar_t)  # 根号alpha累乘
        oneover_sqrta = 1 / torch.sqrt(alpha_t)  # 1 / 根号alpha

        sqrtmab = torch.sqrt(1 - alphabar_t)  # 根号下（1-alpha累乘）
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,  # \alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alphabar_t": alphabar_t,  # \bar{\alpha_t}, \alpha_t累积
            "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}, \alpha_t累积均方根
            "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}, (1 - \alpha_t累积)均方根
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        }

    def forward(self, x):
        """
        训练过程中, 随机选择step和生成噪声
        """
        # 为每个样本随机选择一个step，这有助于模型学习在不同step下的去噪过程。
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        # 随机生成正态分布噪声
        noise = torch.randn_like(x)  # eps ~ N(0, 1)
        # 根据公式计算加噪后的图像x_t
        x_t = (
                # 根据step索引，在末尾添加三个 None 来扩展维度
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise

        )
        # 将unet预测的对应step的正态分布噪声与真实噪声做对比
        return self.loss_mse(noise, self.model(x_t, _ts / self.n_T))

    def sample(self, n_sample, size, device):
        # 随机生成初始噪声图片 x_T ~ N(0, 1)
        x_i = torch.randn(n_sample, *size).to(device)
        for i in range(self.n_T, 0, -1):
            # 归一化的step
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)
            # 在除了最后一个时间步(i == 1)之外的每个step生成一个新的噪声张量z
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            # eps是在step_i时加入到图像中的预测噪声。
            eps = self.model(x_i, t_is)
            # De-noising + Re-noising
            x_i = x_i[:n_sample]
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
        return x_i

class DDIM(DDPM):
    def __init__(self, model, betas, n_T, device):
        super(DDIM, self).__init__(model, betas, n_T, device)

    def ddim_sampling_steps(self, num_steps, eta=0.0):
        """
        用于生成 DDIM 的调度表.
        eta 控制随机性, eta=0 时采样过程是确定性的，eta > 0 时添加一定随机性。

        :param num_steps: 用于 DDIM 的总步数
        :param eta: 采样过程的随机性系数
        """
        c = self.n_T // num_steps
        alphas = self.alphabar_t[1:self.n_T + 1:c]  # 在alphabar_t中按步采样
        alphas_prev = torch.cat([self.alphabar_t[0:1], alphas[:-1]])  # 当前步之前的alphabar_t

        return {
            "alphas": alphas,
            "alphas_prev": alphas_prev,
            "sigmas": eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
        }

    def sample(self, n_sample, size, device, num_steps=50, eta=0.0):
        """
        使用 DDIM 进行采样
        :param num_steps: 生成图像时的步数
        :param eta: 随机性系数 (0 = 确定性采样, 1 = 完全随机采样)
        """
        ddim_steps = self.ddim_sampling_steps(num_steps, eta)
        x_i = torch.randn(n_sample, *size).to(device)  # 初始化噪声图像

        for i in reversed(range(num_steps)):
            # 归一化时间步
            t_is = torch.full((n_sample,), i, dtype=torch.long, device=device)
            t_is = t_is / num_steps

            # 预测噪声
            eps = self.model(x_i, t_is.unsqueeze(-1))

            # 计算确定性或随机性采样
            x_0_pred = (x_i - ddim_steps["sigmas"][i] * eps) / ddim_steps["alphas"][i].sqrt()
            if eta == 0:
                x_i = x_0_pred
            else:
                z = torch.randn_like(x_i) if i > 0 else 0
                x_i = ddim_steps["alphas_prev"][i].sqrt() * x_0_pred + ddim_steps["sigmas"][i] * z

        return x_i

class Unet2(Unet):
    def __init__(self, in_channels, n_feat=256, n_classes=10):
        super().__init__(in_channels, n_feat=n_feat)  # 调用父类构造函数

        # 添加额外的条件嵌入层
        self.conditionembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.conditionembed2 = EmbedFC(n_classes, 1 * n_feat)

    def forward(self, x, c, t):
        '''
        输入加噪图像和对应的时间step，预测反向噪声的正态分布
        :param x: 加噪图像
        :param c: condition向量
        :param t: 对应step
        :return: 正态分布噪声
        '''
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # embed time step
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        cemb1 = self.conditionembed1(c).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.conditionembed2(c).view(-1, self.n_feat, 1, 1)

        # 将上采样输出与step编码相加，输入到下一个上采样层
        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


class DDPM2(DDPM):
    def __init__(self, model, betas, n_T, device, loss_fun=['mse', ]):
        super().__init__(model, betas, n_T, device)
        self.loss_fun = loss_fun

    def forward(self, x, c):
        """
        训练过程中, 随机选择step和生成噪声
        """
        # 随机选择step
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        # 随机生成正态分布噪声
        noise = torch.randn_like(x)  # eps ~ N(0, 1)
        # 加噪后的图像x_t
        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise

        )
        # 将unet预测的对应step的正态分布噪声与真实噪声做对比
        return self.loss_function(noise, self.model(x_t, c, _ts / self.n_T))

    def sample(self, n_sample, c, size, device):
        # 随机生成初始噪声图片 x_T ~ N(0, 1)
        x_i = torch.randn(n_sample, *size).to(device)
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            eps = self.model(x_i, c, t_is)
            x_i = x_i[:n_sample]
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
        return x_i

    def loss_function(self, pred, target):
        total_loss = 0
        if 'mse' in self.loss_fun:
            total_loss += nn.functional.mse_loss(pred, target)
        if 'mae' in self.loss_fun:
            total_loss += nn.functional.l1_loss(pred, target)
        return total_loss

class DDPM3(DDPM2):
    def __init__(self, model, betas, n_T, device, n_classes=10):
        super().__init__(model, betas, n_T, device)
        self.n_classes = n_classes

    def sample(self, n_sample, size, device, c=None, cfg_scale=1.0):
        # 随机生成初始噪声图片 x_T ~ N(0, 1)
        x_i = torch.randn(n_sample, *size).to(device)
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            if c is not None:
                # 有条件输出
                # print(x_i.shape, c.shape, c[0])
                out_cond = self.model(x_i, c, t_is)
                # 无条件输出
                c_uncond = torch.zeros_like(c)
                out_uncond = self.model(x_i, c_uncond, t_is)
                # 应用CFG
                eps = out_uncond + cfg_scale * (out_cond - out_uncond)
            else:
                # 只使用无条件输出
                zero_tensor = torch.zeros((n_sample, self.n_classes))
                eps = self.model(x_i, t_is, zero_tensor)
            x_i = x_i[:n_sample]
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
        return x_i

class ImageGenerator(object):
    def __init__(self, type=None):
        '''
        初始化，定义超参数、数据集、网络结构等
        '''
        self.epoch = 50
        self.sample_num = 100
        self.batch_size = 64
        self.lr = 0.0001
        self.n_T = 400
        self.unconditional_ratio = 0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.type = type
        self.init_dataloader()
        if type is None:
            self.sampler = DDPM(model=Unet(in_channels=1), betas=(1e-4, 0.02), n_T=self.n_T, device=self.device).to(
                self.device)
        elif type == 'classifier_guided':
            self.sampler = DDPM2(model=Unet2(in_channels=1), betas=(1e-4, 0.02), n_T=self.n_T, device=self.device).to(
                self.device)
        elif type == 'classifier_free_guided':
            self.sampler = DDPM3(model=Unet2(in_channels=1), betas=(1e-4, 0.02), n_T=self.n_T, device=self.device).to(
                self.device)
        self.optimizer = optim.Adam(self.sampler.model.parameters(), lr=self.lr)

    def init_dataloader(self):
        '''
        初始化数据集和dataloader
        '''
        # 定义包含随机旋转的转换
        tf = transforms.Compose([
            transforms.RandomRotation(degrees=(-5, 5)),  # 随机旋转 -5 到 5 度
            transforms.ToTensor(),
        ])
        # 初始化训练数据集和 DataLoader
        train_dataset = MNIST('../data/',
                              train=True,
                              download=True,
                              transform=tf)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        # 验证集不需要随机旋转
        val_tf = transforms.Compose([
            transforms.ToTensor(),
        ])
        val_dataset = MNIST('../data/',
                            train=False,
                            download=True,
                            transform=val_tf)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        if not os.path.exists('./model'):
            os.makedirs('./model')
        self.sampler.train()
        print('训练开始!!')
        for epoch in range(self.epoch):
            start_time = time.time()
            self.sampler.model.train()
            loss_mean = 0
            train_dataloader_tqdm = tqdm(self.train_dataloader, desc=f'Epoch {epoch + 1}/{self.epoch}', unit='batch')

            for i, (images, labels) in enumerate(train_dataloader_tqdm):
                images, labels = images.to(self.device), labels.to(self.device)

                if self.type is None:
                    # 使用默认的DDPM训练
                    loss = self.sampler(images)
                elif self.type == 'classifier_guided':
                    # 使用带有条件的DDPM训练
                    labels = F.one_hot(labels, num_classes=10).float()
                    loss = self.sampler(images, labels)
                elif self.type == 'classifier_free_guided':
                    # 使用无条件和有条件的DDPM3训练
                    labels_cond = F.one_hot(labels, num_classes=10).float()
                    # 随机选择一部分样本作为无条件样本
                    is_unconditional = torch.rand(self.batch_size, device=self.device) < self.unconditional_ratio
                    labels_cond = torch.where(is_unconditional.unsqueeze(1), torch.zeros_like(labels_cond), labels_cond)
                    # torch.Size([64, 1, 28, 28]) torch.Size([64, 10])
                    loss = self.sampler(images, labels_cond)

                loss_mean += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 在 batch 级别的进度条中显示当前损失值
                train_dataloader_tqdm.set_postfix(loss=loss.item())

            # 结果
            train_loss = loss_mean / len(self.train_dataloader)
            end_time = time.time()
            print(f'epoch:{epoch}, loss:{train_loss:.4f}, time:{end_time - start_time}')

            # 保存模型
            if epoch % 10 == 0:
                model_path = f'./model/model_epoch_{epoch}.pth'
                torch.save(self.sampler.model.state_dict(), model_path)
                print(f'Model saved to {model_path}')

            # 可视化
            self.visualize_results(epoch)
            print(f'图片已保存')

    @torch.no_grad()
    def visualize_results(self, epoch):
        self.sampler.eval()
        # 保存结果路径
        output_path = 'results'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        tot_num_samples = self.sample_num
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if self.type is None:
            # 无条件生成
            out = self.sampler.sample(tot_num_samples, (1, 28, 28), self.device)
            image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        elif self.type == 'classifier_guided':
            # 条件生成
            labels = F.one_hot(torch.Tensor(np.repeat(np.arange(10), 10)).to(torch.int64), num_classes=10).to(
                self.device).float()
            out = self.sampler.sample(tot_num_samples, labels, (1, 28, 28), self.device)
        elif self.type == 'classifier_free_guided':
            # 条件输出: 100个样本
            labels = F.one_hot(torch.Tensor(np.repeat(np.arange(10), 10)).to(torch.int64), num_classes=10).to(
                self.device).float()
            out_cond = self.sampler.sample(n_sample=tot_num_samples, c=labels, size=(1, 28, 28), device=self.device)
            # 无条件输出: 10个样本
            unconditional_labels = torch.zeros(10, 10).to(self.device).float()
            out_uncond = self.sampler.sample(n_sample=10, c=unconditional_labels, size=(1, 28, 28), device=self.device, cfg_scale=0.0)
            out = torch.cat((out_cond, out_uncond), dim=0)

        save_image(out, os.path.join(output_path, f'{epoch}.jpg'), nrow=image_frame_dim)

    def load_model(self, model_path):
        '''
        从指定路径加载模型参数
        '''
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        self.sampler.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f'Model loaded from {model_path}')


if __name__ == '__main__':
    generator = ImageGenerator()
    # generator.load_model('./model/model_epoch_0.pth')
    # generator.visualize_results(0)
    generator.train()