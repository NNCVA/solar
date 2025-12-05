# 光伏功率预测系统

基于深度学习的光伏功率预测系统，使用过去3天的光伏数据和未来1天的天气预报数据，预测未来1天(96个15分钟数据点)的光伏功率。

## 系统架构

系统由三个核心模块组成：

1. **光伏特征提取模块** ([PVFeatureExtractorWithPDFormer](file://D:\pycharm\py_project\UV3\code(Claude)\models.py#L9-L67))
   - 使用PDFormer（一种先进的时空注意力模型）提取过去3天光伏数据的时序特征
   - 通过1D卷积下采样从288个时间步到96个时间步
   - 输出96条特征向量用于后续融合

2. **天气预报映射模块** ([WeatherForecastMapper](file://D:\pycharm\py_project\UV3\code(Claude)\models.py#L184-L244))
   - 使用LSTM处理24小时天气预报数据
   - 通过转置卷积上采样从24个时间步到96个时间步
   - 将小时级数据映射为15分钟级数据

3. **特征融合预测模块** ([FeatureFusionPredictor](file://D:\pycharm\py_project\UV3\code(Claude)\models.py#L247-L289))
   - 使用多层感知机(MLP)融合光伏特征和天气特征
   - 输出未来一天96个时间点的光伏功率预测

## 文件结构

```
.
├── config.py                    # 配置文件(模型参数、训练参数等)
├── data_loader.py               # 数据加载和预处理模块
├── models.py                    # 模型定义(三个核心模块)
├── train.py                     # 训练脚本
├── predict.py                   # 预测和评估脚本
├── readme.md                    # 使用说明
├── p2_long_knn.csv              # 光伏数据(需准备)
├── nwp.csv                      # 天气预报数据(需准备)
├── saved_models_pd/             # 模型保存目录(自动创建)
│   ├── best_model.pth           # 最佳模型权重
│   ├── scalers.pkl              # 数据标准化器
│   ├── training_curves.png      # 训练曲线
│   ├── predictions.csv          # 测试集预测结果
│   ├── predictions_samples.png  # 样本预测对比图
│   └── error_analysis.png       # 误差分析图
└── model/
    └── traffic_flow_prediction/
        └── PDFormer.py          # PDFormer模型实现
```


## 数据格式要求

### 1. 光伏数据 (p2_long_knn.csv)

**采样频率**: 15分钟  
**时间范围**: 2023-01-01 到 2024-05-30  
**每天记录数**: 96条

**必需列**:
- `date`: 时间戳 (格式: YYYY-MM-DD HH:MM:SS)
- `GHI`: 全球水平辐射
- `DNI`: 直接正射辐射
- `DHI`: 散射辐射
- `WS`: 风速
- `WD`: 风向
- `Temp`: 温度
- `Humidity`: 湿度
- `Pressure`: 气压
- `PV`: 光伏功率(目标变量)

**可选列** (如果没有会自动生成):
- `minute_of_day`: 一天中的分钟数(0-1439)
- `day_of_week`: 星期几(0-6)

### 2. 天气预报数据 (nwp.csv)

**采样频率**: 1小时  
**每天记录数**: 24条

**必需列**:
- `date`: 时间戳 (格式: YYYY-MM-DD HH:MM:SS)
- `Temp`: 预报温度
- `Humidity`: 预报湿度
- `GHI`: 预报全球水平辐射
- `WS`: 预报风速
- `precip`: 预报降水量

## 安装依赖

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm
```


## 使用方法

### 1. 配置参数

在 [config.py](file://D:\pycharm\py_project\UV3\code(Claude)\config.py) 中修改配置:

```python
class Config:
    # 数据路径
    PV_DATA_PATH = 'p2_long_knn.csv'
    WEATHER_DATA_PATH = 'nwp.csv'
    
    # 模型参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    
    # 设备选择
    DEVICE = 'cuda'  # 或 'cpu'
```


### 2. 训练模型

```bash
python train.py
```


训练过程会:
- 自动加载和预处理数据
- 划分训练集、验证集和测试集(7:1.5:1.5)
- 进行数据归一化并保存标准化器
- 训练模型并保存最佳模型
- 生成训练曲线图

### 3. 评估模型

```bash
python predict.py
```


评估过程会:
- 加载训练好的模型和标准化器
- 在测试集上生成预测
- 计算评估指标(MAE, RMSE, MAPE, R²)
- 生成可视化结果

### 4. 自定义预测

```python
import torch
from config import Config
from models import PVPredictionModel
from data_loader import PVDataProcessor

# 加载配置和模型
config = Config()
model = PVPredictionModel(config)
checkpoint = torch.load('saved_models_pd/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 加载标准化器
processor = PVDataProcessor(config)
processor.load_scalers('saved_models_pd')

# 准备输入数据
# pv_data: (batch, 288, 13) - 过去3天的光伏数据
# weather_data: (batch, 24, 5) - 未来1天的天气预报

# 归一化
pv_data_normalized = processor.pv_scaler.transform(pv_data)
weather_data_normalized = processor.weather_scaler.transform(weather_data)

# 转换为tensor
pv_tensor = torch.FloatTensor(pv_data_normalized).unsqueeze(0)
weather_tensor = torch.FloatTensor(weather_data_normalized).unsqueeze(0)

# 预测
with torch.no_grad():
    predictions = model(pv_tensor, weather_tensor)

# 反归一化
predictions = processor.target_scaler.inverse_transform(
    predictions.numpy().reshape(-1, 1)
).reshape(96)
```


## 模型参数说明

### 光伏特征提取模块（PDFormer）
- [PD_D_MODEL](file://D:\pycharm\py_project\UV3\code(Claude)\config.py#L38-L38): PDFormer内部嵌入维度(默认64)
- [PD_N_HEADS](file://D:\pycharm\py_project\UV3\code(Claude)\config.py#L39-L39): 多头注意力头数(默认4)
- [PD_N_LAYERS](file://D:\pycharm\py_project\UV3\code(Claude)\config.py#L40-L40): 编码器层数(默认2)
- [PD_DROPOUT](file://D:\pycharm\py_project\UV3\code(Claude)\config.py#L41-L41): Dropout比率(默认0.1)
- [PD_S_ATTN_SIZE](file://D:\pycharm\py_project\UV3\code(Claude)\config.py#L42-L42): 空间/模式注意力窗口大小(默认3)

### 天气预报映射模块
- [WEATHER_HIDDEN_DIM](file://D:\pycharm\py_project\UV3\code(Claude)\config.py#L47-L47): LSTM隐藏层维度(默认64)
- [WEATHER_NUM_LAYERS](file://D:\pycharm\py_project\UV3\code(Claude)\config.py#L48-L48): LSTM层数(默认2)
- [WEATHER_OUTPUT_DIM](file://D:\pycharm\py_project\UV3\code(Claude)\config.py#L49-L49): 输出特征维度(默认64)
- [UPSAMPLE_FACTOR](file://D:\pycharm\py_project\UV3\code(Claude)\config.py#L50-L50): 上采样因子(默认4，即24→96)

### 特征融合预测模块
- [FUSION_HIDDEN_DIMS](file://D:\pycharm\py_project\UV3\code(Claude)\config.py#L53-L53): MLP隐藏层维度列表(默认[256, 128, 64])
- [FUSION_DROPOUT](file://D:\pycharm\py_project\UV3\code(Claude)\config.py#L54-L54): Dropout比率(默认0.2)

## 关键特性

### 数据预处理
- **缺失值处理**: 线性插值 + 均值填充
- **时间特征编码**: 使用sin/cos编码保持周期性
- **数据归一化**: StandardScaler标准化
- **序列对齐**: 确保光伏数据和天气预报的时间对齐

### 模型特性
- **先进时空建模**: 使用PDFormer替代传统LSTM，具有更强的时空特征提取能力
- **智能上/下采样**: 卷积网络实现时间分辨率转换
- **特征融合**: MLP深度融合多源特征
- **批归一化**: 提高训练稳定性
- **Dropout**: 防止过拟合

### 训练策略
- **学习率调度**: ReduceLROnPlateau自适应调整
- **梯度裁剪**: 防止梯度爆炸
- **早停机制**: 避免过拟合
- **模型检查点**: 自动保存最佳模型

## 评估指标

- **MAE** (Mean Absolute Error): 平均绝对误差
- **RMSE** (Root Mean Square Error): 均方根误差
- **MAPE** (Mean Absolute Percentage Error): 平均绝对百分比误差
- **R²** (R-squared): 决定系数

## 注意事项

1. **数据质量**: 确保数据完整性,缺失值过多会影响预测精度
2. **时间对齐**: 光伏数据和天气预报必须时间对齐
3. **GPU使用**: 建议使用GPU训练以加快速度
4. **内存管理**: 大批量训练时注意内存使用
5. **参数调优**: 根据实际数据调整模型参数

## 常见问题

### Q: 数据格式错误怎么办?
A: 检查CSV文件的列名和日期格式,确保与要求一致。

### Q: 训练过程中loss不下降?
A: 尝试降低学习率、增加批量大小或检查数据质量。

### Q: 预测结果全为0或异常?
A: 检查数据归一化是否正确,确保使用了正确的标准化器。

### Q: 如何提高预测精度?
A: 可以尝试:
- 增加训练数据量
- 调整模型参数(增加隐藏层维度)
- 添加更多特征
- 使用集成学习方法

## 扩展功能

系统设计灵活,便于扩展:

1. **更换特征提取模块**: 可以使用其他先进的时序模型替代PDFormer
2. **添加注意力机制**: 在融合模块中加入注意力权重
3. **多步预测**: 扩展为预测多天的光伏功率
4. **不确定性估计**: 使用贝叶斯神经网络或集成方法
5. **在线学习**: 实现增量学习机制

## 许可证

MIT License

## 联系方式

如有问题或建议,请提issue或联系开发者。