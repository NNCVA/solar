"""
预测脚本 - 负责模型预测和评估
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from config import Config
from models import PVPredictionModel
from data_loader import create_dataloaders


class Predictor:
    """预测器类"""
    
    def __init__(self, model, test_loader, processor, config):
        self.model = model
        self.test_loader = test_loader
        self.processor = processor
        self.config = config
        
        # 设置设备
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self):
        """对测试集进行预测"""
        all_predictions = []
        all_targets = []
        all_dates = []
        
        print("Generating predictions...")
        with torch.no_grad():
            # 遍历测试集索引，而不是直接遍历DataLoader
            for idx in tqdm(range(len(self.test_loader.dataset))):
                # 获取单个样本
                sample = self.test_loader.dataset[idx]
                pv_data = sample['pv_features'].unsqueeze(0).to(self.device)
                weather_data = sample['weather_features'].unsqueeze(0).to(self.device)
                targets = sample['targets'].unsqueeze(0).to(self.device)

                # 获取日期（从dataset的dates列表中）
                date = self.test_loader.dataset.dates[idx]

                # 前向传播
                predictions = self.model(pv_data, weather_data)

                # 收集结果
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_dates.append(date)
        
        # 合并所有批次
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # 反归一化
        pred_shape = all_predictions.shape
        all_predictions = self.processor.target_scaler.inverse_transform(
            all_predictions.reshape(-1, 1)
        ).reshape(pred_shape)
        all_targets = self.processor.target_scaler.inverse_transform(
            all_targets.reshape(-1, 1)
        ).reshape(pred_shape)
        
        return all_predictions, all_targets, all_dates
    
    def evaluate(self, predictions, targets):
        """计算评估指标"""
        # 基本指标
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100
        
        # R² 分数
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
        
        return metrics
    
    def plot_predictions(self, predictions, targets, dates, num_samples=5):
        """可视化预测结果"""
        print(f"\nPlotting {num_samples} sample predictions...")
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        # 随机选择样本
        indices = np.random.choice(len(predictions), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            pred = predictions[idx].flatten()
            target = targets[idx].flatten()
            date = dates[idx]
            
            # 创建时间轴（15分钟间隔）
            time_steps = np.arange(96) * 15 / 60  # 转换为小时
            
            axes[i].plot(time_steps, target, label='Actual', linewidth=2, alpha=0.7)
            axes[i].plot(time_steps, pred, label='Predicted', linewidth=2, alpha=0.7)
            axes[i].set_xlabel('Hour of Day')
            axes[i].set_ylabel('PV Power')
            axes[i].set_title(f'Date: {date}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        os.makedirs(self.config.MODEL_SAVE_PATH, exist_ok=True)
        save_path = os.path.join(self.config.MODEL_SAVE_PATH, 'predictions_samples.png')
        plt.savefig(save_path, dpi=150)
        print(f"Sample predictions saved to {save_path}")
        plt.close()
    
    def plot_error_distribution(self, predictions, targets):
        """绘制误差分布"""
        errors = predictions.flatten() - targets.flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 误差直方图
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Prediction Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Error Distribution')
        axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0].grid(True, alpha=0.3)
        
        # 散点图
        axes[1].scatter(targets.flatten(), predictions.flatten(), alpha=0.3, s=1)
        axes[1].plot([targets.min(), targets.max()], 
                     [targets.min(), targets.max()], 
                     'r--', linewidth=2, label='Perfect Prediction')
        axes[1].set_xlabel('Actual PV Power')
        axes[1].set_ylabel('Predicted PV Power')
        axes[1].set_title('Predicted vs Actual')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.config.MODEL_SAVE_PATH, 'error_analysis.png')
        plt.savefig(save_path, dpi=150)
        print(f"Error analysis saved to {save_path}")
        plt.close()
    
    def save_predictions(self, predictions, targets, dates):
        """保存预测结果到CSV"""
        # 准备数据
        results = []
        for i, date in enumerate(dates):
            for j in range(96):
                results.append({
                    'date': date,
                    'time_step': j,
                    'hour': j // 4,
                    'minute': (j % 4) * 15,
                    'actual': targets[i, j, 0],
                    'predicted': predictions[i, j, 0],
                    'error': predictions[i, j, 0] - targets[i, j, 0]
                })
        
        df = pd.DataFrame(results)
        
        # 保存
        save_path = os.path.join(self.config.MODEL_SAVE_PATH, 'predictions.csv')
        df.to_csv(save_path, index=False)
        print(f"Predictions saved to {save_path}")
    
    def run_evaluation(self):
        """运行完整的评估流程"""
        # 生成预测
        predictions, targets, dates = self.predict()
        
        # 计算指标
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        
        metrics = self.evaluate(predictions, targets)
        for metric, value in metrics.items():
            if metric == 'MAPE':
                print(f"{metric}: {value:.2f}%")
            else:
                print(f"{metric}: {value:.4f}")
        
        print("="*50)
        
        # 可视化
        self.plot_predictions(predictions, targets, dates, num_samples=5)
        self.plot_error_distribution(predictions, targets)
        
        # 保存结果
        self.save_predictions(predictions, targets, dates)
        
        return metrics


def load_trained_model(config):
    """加载训练好的模型"""
    # 创建模型
    model = PVPredictionModel(config)
    
    # 加载检查点
    checkpoint_path = os.path.join(config.MODEL_SAVE_PATH, config.SAVE_MODEL)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
    
    return model


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # 创建配置
    config = Config()
    
    print("Loading data...")
    # 创建数据加载器
    _, _, test_loader, processor = create_dataloaders(config)
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # 加载训练好的模型
    print("\nLoading trained model...")
    model = load_trained_model(config)
    
    # 创建预测器
    predictor = Predictor(model, test_loader, processor, config)
    
    # 运行评估
    print("\nRunning evaluation on test set...")
    metrics = predictor.run_evaluation()
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
