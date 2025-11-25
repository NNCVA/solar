"""
数据处理模块 - 负责数据加载、预处理和Dataset创建
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
from config import Config


class PVDataProcessor:
    """光伏数据预处理器"""
    
    def __init__(self, config):
        self.config = config
        self.pv_scaler = StandardScaler()
        self.weather_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def load_data(self):
        """加载光伏数据和天气预报数据"""
        # 加载光伏数据
        pv_data = pd.read_csv(self.config.PV_DATA_PATH)
        pv_data['date'] = pd.to_datetime(pv_data['date'])
        
        # 加载天气预报数据
        weather_data = pd.read_csv(self.config.WEATHER_DATA_PATH)
        weather_data['date'] = pd.to_datetime(weather_data['date'])
        
        return pv_data, weather_data

    def handle_missing_values(self, df):
        """处理缺失值 - 使用线性插值"""
        # 对数值列进行插值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
        
        # 如果仍有缺失值，使用均值填充
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        return df
    
    def add_time_features(self, df):
        """添加时间特征的周期性编码"""
        # 分钟的周期性编码（0-1439分钟）
        if 'minute_of_day' not in df.columns:
            df['minute_of_day'] = df['date'].dt.hour * 60 + df['date'].dt.minute
        
        # 星期的周期性编码（0-6）
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['date'].dt.dayofweek
        
        # 使用sin/cos编码保持周期性
        df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / 1440)
        df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / 1440)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def prepare_sequences(self, pv_data, weather_data):
        """
        准备训练序列
        每个样本包含：
        - 过去3天的光伏数据（288条记录）
        - 未来1天的天气预报（24条记录）
        - 未来1天的PV目标值（96条记录）
        """
        # 处理缺失值
        pv_data = self.handle_missing_values(pv_data)
        weather_data = self.handle_missing_values(weather_data)
        
        # 添加时间特征
        pv_data = self.add_time_features(pv_data)
        
        # 按日期排序
        pv_data = pv_data.sort_values('date').reset_index(drop=True)
        weather_data = weather_data.sort_values('date').reset_index(drop=True)
        
        # 获取唯一日期
        pv_dates = pv_data['date'].dt.date.unique()
        weather_dates = weather_data['date'].dt.date.unique()
        
        # 找到共同日期范围
        common_dates = sorted(set(pv_dates) & set(weather_dates))
        
        sequences = []
        lookback = self.config.LOOKBACK_DAYS
        
        # 遍历每个可能的预测日
        for i in range(lookback, len(common_dates)):
            # 获取过去3天的日期
            past_dates = common_dates[i-lookback:i]
            # 获取未来1天的日期
            future_date = common_dates[i]
            
            # 提取过去3天的光伏数据（288条记录）
            past_pv = pv_data[pv_data['date'].dt.date.isin(past_dates)]
            
            # 提取未来1天的天气预报（24条记录）
            future_weather = weather_data[weather_data['date'].dt.date == future_date]
            
            # 提取未来1天的PV目标值（96条记录）
            future_pv = pv_data[pv_data['date'].dt.date == future_date]
            
            # 验证数据完整性
            if (len(past_pv) == self.config.PV_SAMPLES_PER_DAY * lookback and
                len(future_weather) == self.config.WEATHER_SAMPLES_PER_DAY and
                len(future_pv) == self.config.PV_SAMPLES_PER_DAY):
                
                sequences.append({
                    'past_pv': past_pv,
                    'future_weather': future_weather,
                    'future_pv': future_pv,
                    'date': future_date
                })
        
        return sequences
    
    def fit_scalers(self, sequences):
        """拟合标准化器"""
        # 收集所有光伏特征数据
        all_pv_features = []
        all_weather_features = []
        all_targets = []
        
        for seq in sequences:
            # 光伏特征（包含时间编码）
            pv_features = seq['past_pv'][self.config.PV_FEATURE_COLS + 
                                         ['minute_sin', 'minute_cos', 'day_sin', 'day_cos']].values
            all_pv_features.append(pv_features)
            
            # 天气特征
            weather_features = seq['future_weather'][self.config.WEATHER_FEATURE_COLS].values
            all_weather_features.append(weather_features)
            
            # 目标值
            targets = seq['future_pv'][self.config.TARGET_COL].values
            all_targets.append(targets)
        
        # 拟合标准化器
        all_pv_features = np.vstack(all_pv_features)
        all_weather_features = np.vstack(all_weather_features)
        all_targets = np.concatenate(all_targets).reshape(-1, 1)
        
        self.pv_scaler.fit(all_pv_features)
        self.weather_scaler.fit(all_weather_features)
        self.target_scaler.fit(all_targets)
    
    def save_scalers(self, path):
        """保存标准化器"""
        with open(f'{path}/scalers.pkl', 'wb') as f:
            pickle.dump({
                'pv_scaler': self.pv_scaler,
                'weather_scaler': self.weather_scaler,
                'target_scaler': self.target_scaler
            }, f)
    
    def load_scalers(self, path):
        """加载标准化器"""
        with open(f'{path}/scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
            self.pv_scaler = scalers['pv_scaler']
            self.weather_scaler = scalers['weather_scaler']
            self.target_scaler = scalers['target_scaler']


class PVDataset(Dataset):
    """光伏预测数据集"""
    
    def __init__(self, sequences, processor, config):
        self.sequences = sequences
        self.processor = processor
        self.config = config
        # 提取日期信息到单独的列表
        self.dates = [seq['date'] for seq in sequences]

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # 提取光伏特征（288 x 13）
        pv_features = seq['past_pv'][self.config.PV_FEATURE_COLS + 
                                      ['minute_sin', 'minute_cos', 'day_sin', 'day_cos']].values
        pv_features = self.processor.pv_scaler.transform(pv_features)
        
        # 提取天气特征（24 x 5）
        weather_features = seq['future_weather'][self.config.WEATHER_FEATURE_COLS].values
        weather_features = self.processor.weather_scaler.transform(weather_features)
        
        # 提取目标值（96 x 1）
        targets = seq['future_pv'][self.config.TARGET_COL].values.reshape(-1, 1)
        targets = self.processor.target_scaler.transform(targets)
        
        return {
            'pv_features': torch.FloatTensor(pv_features),
            'weather_features': torch.FloatTensor(weather_features),
            'targets': torch.FloatTensor(targets),
            # 'date': seq['date']
        }


def create_dataloaders(config):
    """创建训练、验证和测试数据加载器"""
    # 初始化数据处理器
    processor = PVDataProcessor(config)
    
    # 加载数据
    pv_data, weather_data = processor.load_data()
    
    # 准备序列
    sequences = processor.prepare_sequences(pv_data, weather_data)
    
    # 划分数据集
    n_total = len(sequences)
    n_train = int(n_total * config.TRAIN_RATIO)
    n_val = int(n_total * config.VAL_RATIO)
    
    train_sequences = sequences[:n_train]
    val_sequences = sequences[n_train:n_train+n_val]
    test_sequences = sequences[n_train+n_val:]
    
    # 拟合标准化器（仅使用训练集）
    processor.fit_scalers(train_sequences)
    
    # 创建数据集
    train_dataset = PVDataset(train_sequences, processor, config)
    val_dataset = PVDataset(val_sequences, processor, config)
    test_dataset = PVDataset(test_sequences, processor, config)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, processor
