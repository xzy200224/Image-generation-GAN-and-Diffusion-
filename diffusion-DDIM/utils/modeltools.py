import torch
import math
from pathlib import Path
from typing import Union, Dict
import yaml
from tqdm import tqdm


def load_yaml(yml_path: Union[Path, str], encoding="utf-8"):
    """Load yaml 配准文件."""
    if isinstance(yml_path, str):
        yml_path = Path(yml_path)
    with yml_path.open('r', encoding=encoding) as f:
        cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        return cfg


def train_one_epoch(trainer, loader, optimizer, device, epoch):
    """训练一个 epoch。"""
    trainer.train()
    total_loss, total_num = 0., 0

    with tqdm(loader, dynamic_ncols=True, colour="#ff924a") as data:
        data.set_description(f"Epoch: {epoch}")
        for images, _ in data:
            optimizer.zero_grad()
            x_0 = images.to(device)
            loss = trainer(x_0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_num += x_0.shape[0]
            # 更新进度条显示的信息，包括当前 epoch 和实时的平均训练损失。
            avg_loss = total_loss / total_num
            data.set_postfix(ordered_dict={
                "train_loss": f"{avg_loss:.4f}",
            })

    return avg_loss

class ModelCheckpoint:
    def __init__(self, filepath: str = 'checkpoint.pth', monitor: str = 'val_loss',
                 mode: str = 'min', save_best: bool = False, save_freq: int = 2, early_stopping: int = 0):
        """
        在训练过程中自动保存检查点。

        参数:
            filepath: 文件名或文件夹名，模型保存位置
            monitor: 要监控的指标，只有当传递的是`dict`时才会生效。
            mode: 监控指标的模式，'min'或'max'。
            save_best_only: 是否只保存具有最佳指标的模型。
            save_freq: 保存频率，仅在`save_best_only=False`时有效。
            early_stopping: 提前终止训练的容忍次数，如果为0则不提前终止训练，大于0时是保存具有最佳指标的模型。
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_freq = save_freq
        self.__times = 1
        self.__patience = 0
        self.__value = -math.inf if mode == 'max' else math.inf
        self.early_stopping = early_stopping
        if self.early_stopping > 0:
            self.save_best = True
        else:
            self.save_best = save_best

    @staticmethod
    def save(filepath: str, times: int = None, **kwargs):
        """
        保存模型。
        参数:
            filepath: 文件名或文件夹名，需要保存的位置。如果是文件夹，
                保存的检查点数量可能超过一个。
            times: 当前保存次数，仅在保存路径为文件夹时用于命名文件。
            kwargs: 所有需要保存的内容。
        """
        path = Path(filepath)

        if path.is_dir():
            if not path.exists():
                path.mkdir(parents=True)
            path = path.joinpath(f'checkpoint-{times}.pth')
        else:
            # 如果 path 不是文件夹路径
            stem = path.stem  # 获取文件名
            suffix = path.suffix  # 获取扩展名
            # 修改文件名为 '文件名_{times}.pth'
            path = path.with_name(f'{stem}_{times}{suffix}')
        torch.save(kwargs, str(path))

    def state_dict(self):
        """
        返回保存状态以便下次加载恢复。
        """
        return {
            'filepath': self.filepath,
            'monitor': self.monitor,
            'save_best_only': self.save_best,
            'mode': self.mode,
            'save_freq': self.save_freq,
            'times': self.__times,
            'value': self.__value
        }

    def load_state_dict(self, state_dict: dict):
        """
        加载状态
        """
        self.filepath = state_dict['filepath']
        self.monitor = state_dict['monitor']
        self.save_best = state_dict['save_best_only']
        self.mode = state_dict['mode']
        self.save_freq = state_dict['save_freq']
        self.__times = state_dict['times']
        self.__value = state_dict['value']

    def reset(self):
        """
        重置计数次数
        """
        self.__times = 1
        self.__patience = 0

    def step(self, metrics: Union[Dict, int, float], **kwargs):
        """
        参数:
            metrics: 包含`monitor`的字典或一个标量
            kwargs: 所有需要保存的内容。
        """
        # 字典则取出验证集loss
        if isinstance(metrics, dict):
            metrics = metrics[self.monitor]

        if self.save_best:
            # 根据效果保存
            if (self.mode == 'min' and metrics <= self.__value) or (
                    self.mode == 'max' and metrics >= self.__value):
                self.__value = metrics
                self.__patience = 0
                self.save(self.filepath, self.__times, **kwargs)
            else:
                self.__patience += 1
        else:
            # 根据频率保存
            if self.__times % self.save_freq == 0:
                self.save(self.filepath, self.__times, **kwargs)
        # 早停
        if self.early_stopping > 0 and self.__patience >= self.early_stopping:
            return False

        self.__times += 1
        return True
