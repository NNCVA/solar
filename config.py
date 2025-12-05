"""
配置文件 - 包含所有模型和训练参数
"""


class Config:
    # =================================================
    # 数据路径配置
    PV_DATA_PATH = 'p2_long_knn.csv'  # 光伏数据路径
    WEATHER_DATA_PATH = 'nwp.csv'  # 天气预报数据路径
    MODEL_SAVE_PATH = 'saved_models_pd/'  # 模型保存路径

    # =================================================
    # 数据参数
    LOOKBACK_DAYS = 3  # 使用过去3天的数据
    PREDICT_DAYS = 1  # 预测未来1天
    PV_SAMPLES_PER_DAY = 96  # 每天96个15分钟采样点
    WEATHER_SAMPLES_PER_DAY = 24  # 天气预报每天24个小时采样点

    # 光伏数据特征列（9个特征）
    PV_FEATURE_COLS = ['GHI', 'DNI', 'DHI', 'WS', 'WD', 'Temp', 'Humidity', 'Pressure', 'PV']

    # 天气预报特征列（5个特征）
    WEATHER_FEATURE_COLS = ['Temp', 'Humidity', 'GHI', 'WS', 'precip']

    # 时间特征列
    TIME_FEATURE_COLS = ['minute_of_day', 'day_of_week']

    # 目标列
    TARGET_COL = 'PV'

    # =================================================
    # 模型参数
    # 1. 光伏特征提取模块
    PV_HIDDEN_DIM = 128  # LSTM隐藏层维度
    PV_NUM_LAYERS = 2  # LSTM层数
    PV_OUTPUT_DIM = 64  # 输出特征维度

    # PDFormer参数
    PD_D_MODEL = 64  # PDFormer 的嵌入维度
    PD_S_ATTN_SIZE = 3  # 空间注意力大小
    PD_T_ATTN_SIZE = 3  # 时间注意力大小
    PD_GEO_NUM_HEADS = 4  # 地理注意力头数
    PD_SEM_NUM_HEADS = 2  # 语义注意力头数
    PD_T_NUM_HEADS = 2  # 时间注意力头数
    PD_ENC_DEPTH = 3  # PDFormer 的编码层数

    # 2. 天气预报映射模块
    WEATHER_HIDDEN_DIM = 64  # LSTM隐藏层维度
    WEATHER_NUM_LAYERS = 2  # LSTM层数
    WEATHER_OUTPUT_DIM = 64  # 输出特征维度
    UPSAMPLE_FACTOR = 4  # 上采样因子（24->96）

    # 3. 特征融合预测模块
    FUSION_HIDDEN_DIMS = [256, 128, 64]  # MLP隐藏层维度
    FUSION_DROPOUT = 0.2  # Dropout比率

    # =================================================
    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    TRAIN_RATIO = 0.7  # 训练集比例
    VAL_RATIO = 0.15  # 验证集比例

    # 优化器参数
    WEIGHT_DECAY = 1e-5

    # 学习率调度器参数
    LR_SCHEDULER_PATIENCE = 10
    LR_SCHEDULER_FACTOR = 0.5

    # 早停参数
    EARLY_STOPPING_PATIENCE = 20

    # 随机种子
    RANDOM_SEED = 42

    # 设备配置
    DEVICE = 'cuda'  # 'cuda' or 'cpu'

    # 最佳模型保存
    SAVE_MODEL = 'best_model.pth'
