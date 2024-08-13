from typing import Optional
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
from torchvision.utils import save_image


def show_images(dataloader, num_images=10, mode='gray'):
    """
    显示 dataloader 中的前 num_images 张图片。

    :param dataloader: 数据加载器，包含图像及其标签。
    :param num_images: 要显示的图像数量，默认为 10。
    :param mode: 图像显示模式，默认为 'gray'，也可以是 'color'。
    """
    for i, (images, labels) in enumerate(dataloader):
        if i == 0:  # 只显示第一个 batch 中的图片
            for j in range(min(num_images, len(images))):
                plt.figure()
                plt.imshow(images[j].squeeze(), cmap=mode)  # 显示灰度或彩色图像
                plt.title(f"Label: {labels[j]}")
                plt.axis('off')
            break
    plt.show()

def save_image(images: torch.Tensor, nrow: int = 8, show: bool = True, save_path: Optional[str] = None,
               format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """
    将所有图像拼接成一张图片。

    参数:
        images: 形状为 (batch_size, channels, height, width) 的张量。
        nrow: 决定每行的图片数量。默认值为 `8`。
        show: 是否在拼接后显示图像。默认值为 `True`。
        path: 保存图像的路径。如果为 None（默认），则不保存图像。
        format: 图像格式。你可以通过运行 `python3 -m PIL` 来打印可用格式的集合。
        to_grayscale: 将 PIL 图像转换为灰度图像。默认值为 `False`。
        **kwargs: 传递给 `torchvision.utils.make_grid` 的其他参数。

    返回:
        拼接后的图像，一个形状为 (height, width, channels) 的张量。
    """
    images = images * 0.5 + 0.5
    grid = make_grid(images, nrow=nrow, **kwargs)  # (channels, height, width)
    #  (height, width, channels)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if save_path is not None:
        im.save(save_path, format=format)
    if show:
        im.show()
    return grid


def save_sample_image(images: torch.Tensor, show: bool = True, save_path: Optional[str] = None,
                      format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """
    将所有图像（包括中间过程）拼接成一张图片。

    参数:
        images: 包含中间过程的图像，
            形状为 (batch_size, sample, channels, height, width) 的张量。
        show: 是否在拼接后显示图像。默认值为 `True`。
        path: 保存图像的路径。如果为 None（默认），则不保存图像。
        format: 图像格式。你可以通过运行 `python3 -m PIL` 来打印可用格式的集合。
        to_grayscale: 将 PIL 图像转换为灰度图像。默认值为 `False`。
        **kwargs: 传递给 `torchvision.utils.make_grid` 的其他参数。

    返回:
        拼接后的图像，一个形状为 (height, width, channels) 的张量。
    """
    images = images * 0.5 + 0.5

    grid = []
    for i in range(images.shape[0]):
        # 对于批次中的每个样本，将所有中间过程图像拼接成一行
        t = make_grid(images[i], nrow=images.shape[1], **kwargs)  # (channels, height, width)
        grid.append(t)
    # 将所有拼接好的图像堆叠成一个张量
    grid = torch.stack(grid, dim=0)  # (batch_size, channels, height, width)
    grid = make_grid(grid, nrow=1, **kwargs)  # concat all batch images in a different row, (channels, height, width)
    #  (height, width, channels)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if save_path is not None:
        im.save(save_path, format=format)
    if show:
        im.show()
    return grid
