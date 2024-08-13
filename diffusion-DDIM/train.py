from dataset import create_dataset
from model.UNet import UNet
from utils.engine import GaussianDiffusionTrainer
from utils.modeltools import train_one_epoch, load_yaml
import torch
from utils.modeltools import ModelCheckpoint
import os

def create_dataloader(data: str='mnist', **kwargs):
    # 数据集
    if data == 'mnist':
        dataset_params = {'dataset': 'mnist', 'train': True, 'data_path': './data/', 'download': True,
                   'image_size': [28, 28], 'batch_size': 32, 'drop_last': True, 'num_workers': 4}
    elif data == "cifar":
        dataset_params = {'dataset': 'cifar', 'train': True, 'data_path': './data/', 'download': True,
                   'image_size': [32, 32], 'mode': 'RGB', 'suffix': ['png', 'jpg'], 'batch_size': 32,
                   'shuffle': True, 'drop_last': True, 'pin_memory': True, 'num_workers': 4}
    dataloader = create_dataset(**dataset_params)
    return dataloader

def train(dataloader, model_params, diffusion_params, optimizer_params, save_params,  config=None):
    start_epoch = 1
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型
    model = UNet(**model_params).to(device)
    trainer = GaussianDiffusionTrainer(model, **diffusion_params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=optimizer_params['lr'], weight_decay=optimizer_params['weight_decay'])
    # 保存
    model_checkpoint = ModelCheckpoint(**save_params)
    # 加载过去参数
    if os.path.exists(save_params['filepath']):
        params = torch.load(save_params['filepath'])
        model.load_state_dict(params["model"])
        optimizer.load_state_dict(params["optimizer"])
        model_checkpoint.load_state_dict(params["model_checkpoint"])
        print(params["model_checkpoint"])
        start_epoch = params["start_epoch"] + 1
    # 训练
    for epoch in range(start_epoch, 200):
        loss = train_one_epoch(trainer, dataloader, optimizer, device, epoch)
        model_checkpoint.step(loss, model=model.state_dict(), optimizer=optimizer.state_dict(),
                              start_epoch=epoch, model_checkpoint=model_checkpoint.state_dict())


if __name__ == "__main__":
    # config = load_yaml("config2.yml", encoding="utf-8")
    model_params = {'in_channels': 1, 'out_channels': 1, 'model_channels': 8, 'attention_resolutions': [16],
                    'num_res_blocks': 2, 'dropout': 0.1, 'channel_mult': [1, 2], 'num_heads': 2}
    diffusion_params = {'T': 1000, 'beta': [0.0001, 0.02], 'loss': ['mae', 'mse']}
    optimizer_params = {'lr': 0.001, 'weight_decay': 1e-4}
    save_params = {'filepath': './checkpoint/mnist_8.pth', 'save_best': True}
    train(model_params=model_params, optimizer_params=optimizer_params, diffusion_params=diffusion_params,
          save_params=save_params, dataloader=create_dataloader())
