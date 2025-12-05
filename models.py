"""
模型定义 - 包含三个核心模块和完整的预测模型
"""
import numpy as np
import torch
import torch.nn as nn
from model.traffic_flow_prediction.PDFormer import PDFormer


class PVFeatureExtractor(nn.Module):
    """
    光伏特征提取模块
    输入: 过去3天的光伏数据 (batch, 288, 13)
    输出: 96条特征向量 (batch, 96, hidden_dim)
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PVFeatureExtractor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 使用LSTM提取时序特征
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 1D卷积进行下采样 (288 -> 96)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 288, 13) - 过去3天的光伏数据
        Returns:
            (batch, 96, output_dim) - 96条特征向量
        """
        # LSTM特征提取
        lstm_out, _ = self.lstm(x)  # (batch, 288, hidden_dim)
        
        # 转置以进行1D卷积 (batch, hidden_dim, 288)
        lstm_out = lstm_out.transpose(1, 2)
        
        # 下采样到96个时间步 (batch, hidden_dim, 96)
        conv_out = self.conv(lstm_out)
        
        # 转置回来 (batch, 96, hidden_dim)
        conv_out = conv_out.transpose(1, 2)
        
        # 输出投影 (batch, 96, output_dim)
        output = self.output_proj(conv_out)
        
        return output


class PVFeatureExtractorWithPDFormer(nn.Module):
    """
    基于 PDFormer 的光伏特征提取模块
    输入: (batch, 288, 13)
    输出: (batch, 96, output_dim)
    """

    def __init__(self, config):
        super(PVFeatureExtractorWithPDFormer, self).__init__()

        self.input_len = config.PV_SAMPLES_PER_DAY * config.LOOKBACK_DAYS  # 288
        self.feature_dim = len(config.PV_FEATURE_COLS) + 4  # 13 (9个特征 + 4个时间编码)
        self.hidden_dim = config.PD_D_MODEL  # 64
        self.output_dim = config.PV_OUTPUT_DIM  # 64

        # --- 1. 构建 PDFormer data_feature ---
        adj_mx = np.zeros((1, 1), dtype=np.float32)
        dtw_matrix = np.zeros((1, 1), dtype=np.float32)
        sd_mx = np.zeros((1, 1), dtype=np.float32)
        sh_mx = np.zeros((1, 1), dtype=np.float32)
        pattern_keys = np.zeros((1, config.PD_S_ATTN_SIZE, self.hidden_dim), dtype=np.float32)

        data_feature = {
            'adj_mx': adj_mx,
            'num_nodes': 1,
            'feature_dim': self.feature_dim,
            'dtw_matrix': dtw_matrix,
            'sd_mx': sd_mx,
            'sh_mx': sh_mx,
            'pattern_keys': pattern_keys,
            # 添加 max_len 参数
            'max_len': 300
        }

        # --- 2. PDFormer 配置 ---
        pd_config = {
            'embed_dim': self.hidden_dim,
            # [关键修改1] output_dim 必须等于特征数量 (13)，不能是 64
            'output_dim': self.feature_dim,
            'input_window': self.input_len,
            'output_window': self.input_len,
            'enc_depth': config.PD_N_LAYERS,
            'geo_num_heads': config.PD_N_HEADS,
            'sem_num_heads': 2,
            't_num_heads': 2,
            's_attn_size': config.PD_S_ATTN_SIZE,
            'drop': config.PD_DROPOUT,
            'device': torch.device(config.DEVICE),
            'add_time_in_day': False,
            'add_day_in_week': False,
            'lape_dim': 8,
            'type_short_path': 'hop',
            # 添加 max_len 参数
            'max_len': 300
        }

        self.pdformer = PDFormer(pd_config, data_feature)

        # [关键修改2] 新增一个投影层，将 PDFormer 输出的 13 维映射回 hidden_dim (64)
        self.feature_proj = nn.Linear(self.feature_dim, self.hidden_dim)

        # --- 3. 后处理模块 ---
        self.conv = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim)
        )

        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(config.PD_DROPOUT)
        )

    # 可以考虑稍微调整一下注释，使逻辑更清晰
    def forward(self, x):
        """
        Args:
            x: (batch, 288, 13) - 输入光伏数据
        Returns:
            output: (batch, 96, 64) - 处理后的特征向量
        """
        # 1. 适配输入格式: (B, T, F) -> (B, T, N=1, F)
        x_reshaped = x.unsqueeze(2)  # (B, 288, 1, 13)
        batch_dict = {'X': x_reshaped}

        # 2. PDFormer 特征提取: (B, 288, 1, 13)
        pd_out = self.pdformer(batch_dict)

        # 3. 维度重塑: (B, 288, 1, 13) -> (B, 288, 13)
        pd_out = pd_out.squeeze(2)

        # 4. 特征维度投影: (B, 288, 13) -> (B, 288, 64)
        pd_out = self.feature_proj(pd_out)

        # 5. 准备卷积下采样: (B, 288, 64) -> (B, 64, 288)
        pd_out = pd_out.permute(0, 2, 1)

        # 6. 卷积下采样: (B, 64, 288) -> (B, 64, 96)
        conv_out = self.conv(pd_out)

        # 7. 维度还原和最终投影: (B, 64, 96) -> (B, 96, 64)
        conv_out = conv_out.transpose(1, 2)
        output = self.output_proj(conv_out)

        return output


class WeatherForecastMapper(nn.Module):
    """
    天气预报映射模块
    输入: 未来1天的天气预报 (batch, 24, 5)
    输出: 96条特征向量 (batch, 96, output_dim)
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, upsample_factor=4):
        super(WeatherForecastMapper, self).__init__()
        
        self.upsample_factor = upsample_factor
        
        # LSTM提取天气特征
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 上采样网络 (24 -> 96)
        # 使用转置卷积进行上采样
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 24, 5) - 未来1天的天气预报
        Returns:
            (batch, 96, output_dim) - 96条特征向量
        """
        # LSTM特征提取
        lstm_out, _ = self.lstm(x)  # (batch, 24, hidden_dim)
        
        # 转置以进行上采样 (batch, hidden_dim, 24)
        lstm_out = lstm_out.transpose(1, 2)
        
        # 上采样到96个时间步 (batch, hidden_dim, 96)
        upsampled = self.upsample(lstm_out)
        
        # 转置回来 (batch, 96, hidden_dim)
        upsampled = upsampled.transpose(1, 2)
        
        # 输出投影 (batch, 96, output_dim)
        output = self.output_proj(upsampled)
        
        return output


class FeatureFusionPredictor(nn.Module):
    """
    特征融合预测模块
    输入: 光伏特征 + 天气特征 (batch, 96, pv_dim + weather_dim)
    输出: 未来1天的PV预测 (batch, 96, 1)
    """
    
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super(FeatureFusionPredictor, self).__init__()
        
        # 构建MLP网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(96),  # 对每个时间步进行归一化
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, pv_features, weather_features):
        """
        Args:
            pv_features: (batch, 96, pv_dim) - 光伏特征
            weather_features: (batch, 96, weather_dim) - 天气特征
        Returns:
            (batch, 96, 1) - PV预测值
        """
        # 拼接特征 (batch, 96, pv_dim + weather_dim)
        fused_features = torch.cat([pv_features, weather_features], dim=-1)
        
        # MLP预测 (batch, 96, 1)
        output = self.mlp(fused_features)
        
        return output


class PVPredictionModel(nn.Module):
    """
    完整的光伏功率预测模型
    整合三个子模块：光伏特征提取、天气预报映射、特征融合预测
    """
    
    def __init__(self, config):
        super(PVPredictionModel, self).__init__()
        
        # 光伏特征提取模块
        # 输入维度：9个光伏特征 + 4个时间编码特征 = 13
        # 使用新的 PVFeatureExtractorWithPDFormer 替换原有的 LSTM 实现
        self.pv_extractor = PVFeatureExtractorWithPDFormer(config)
        
        # 天气预报映射模块
        # 输入维度：5个天气特征
        self.weather_mapper = WeatherForecastMapper(
            input_dim=len(config.WEATHER_FEATURE_COLS),  # 5
            hidden_dim=config.WEATHER_HIDDEN_DIM,
            num_layers=config.WEATHER_NUM_LAYERS,
            output_dim=config.WEATHER_OUTPUT_DIM,
            upsample_factor=config.UPSAMPLE_FACTOR
        )
        
        # 特征融合预测模块
        # 输入维度：光伏特征维度 + 天气特征维度
        self.fusion_predictor = FeatureFusionPredictor(
            input_dim=config.PV_OUTPUT_DIM + config.WEATHER_OUTPUT_DIM,
            hidden_dims=config.FUSION_HIDDEN_DIMS,
            dropout=config.FUSION_DROPOUT
        )
    
    def forward(self, pv_data, weather_data):
        """
        Args:
            pv_data: (batch, 288, 13) - 过去3天的光伏数据
            weather_data: (batch, 24, 5) - 未来1天的天气预报
        Returns:
            (batch, 96, 1) - 未来1天的PV预测值
        """
        # 提取光伏特征 (batch, 96, pv_output_dim)
        pv_features = self.pv_extractor(pv_data)
        
        # 映射天气特征 (batch, 96, weather_output_dim)
        weather_features = self.weather_mapper(weather_data)
        
        # 融合特征并预测 (batch, 96, 1)
        predictions = self.fusion_predictor(pv_features, weather_features)
        
        return predictions
    
    def get_intermediate_features(self, pv_data, weather_data):
        """
        获取中间特征（用于可视化和分析）
        Returns:
            pv_features, weather_features, predictions
        """
        pv_features = self.pv_extractor(pv_data)
        weather_features = self.weather_mapper(weather_data)
        predictions = self.fusion_predictor(pv_features, weather_features)
        
        return pv_features, weather_features, predictions
