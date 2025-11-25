"""
训练脚本 - 负责模型训练和验证
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from models import PVPredictionModel
from data_loader import create_dataloaders


class Trainer:
    """训练器类"""
    
    def __init__(self, model, train_loader, val_loader, processor, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.processor = processor
        self.config = config
        
        # 设置设备
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            verbose=True
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch in pbar:
                # 准备数据
                pv_data = batch['pv_features'].to(self.device)
                weather_data = batch['weather_features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # 前向传播
                predictions = self.model(pv_data, weather_data)
                loss = self.criterion(predictions, targets)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 记录损失
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        return epoch_loss / len(self.train_loader)
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 准备数据
                pv_data = batch['pv_features'].to(self.device)
                weather_data = batch['weather_features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # 前向传播
                predictions = self.model(pv_data, weather_data)
                loss = self.criterion(predictions, targets)
                
                val_loss += loss.item()
                
                # 收集预测和目标值（用于计算指标）
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # 计算平均损失
        avg_val_loss = val_loss / len(self.val_loader)
        
        # 合并所有批次
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # 反归一化
        all_predictions = self.processor.target_scaler.inverse_transform(
            all_predictions.reshape(-1, 1)
        ).reshape(all_predictions.shape)
        all_targets = self.processor.target_scaler.inverse_transform(
            all_targets.reshape(-1, 1)
        ).reshape(all_targets.shape)
        
        # 计算评估指标
        mae = np.mean(np.abs(all_predictions - all_targets))
        rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
        mape = np.mean(np.abs((all_predictions - all_targets) / (all_targets + 1e-8))) * 100
        
        return avg_val_loss, mae, rmse, mape
    
    def train(self):
        """完整的训练流程"""
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, mae, rmse, mape = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(self.config.SAVE_MODEL)
                print(f"✓ Best model saved (val_loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
    
    def save_checkpoint(self, filename):
        """保存模型检查点"""
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, os.path.join(self.config.MODEL_SAVE_PATH, filename))
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        plt.savefig(os.path.join(self.config.MODEL_SAVE_PATH, 'training_curves.png'))
        print(f"Training curves saved to {self.config.MODEL_SAVE_PATH}/training_curves.png")


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # 创建配置
    config = Config()
    
    print("Loading data...")
    # 创建数据加载器
    train_loader, val_loader, test_loader, processor = create_dataloaders(config)
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 保存标准化器
    processor.save_scalers(config.MODEL_SAVE_PATH)
    print(f"Scalers saved to {config.MODEL_SAVE_PATH}/scalers.pkl")
    
    # 创建模型
    print("\nInitializing model...")
    model = PVPredictionModel(config)
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, processor, config)
    
    # 开始训练
    print("\nStarting training...")
    trainer.train()


if __name__ == '__main__':
    main()
