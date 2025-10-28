import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# 1. 配置路径
RAW_DATA_DIR = "/root/autodl-tmp/meituan_comment/data/raw"
DATA_SAVE_PATH = "/root/autodl-tmp/meituan_comment/data_v2"  # 改成新的目录
os.makedirs(DATA_SAVE_PATH, exist_ok=True)

# 2. 从本地文件加载数据集
def load_local_dataset(raw_dir):
    files = os.listdir(raw_dir)

    parquet_files = [f for f in files if f.endswith('.parquet')]
    if parquet_files:
        file_path = os.path.join(raw_dir, parquet_files[0])
        print(f"加载parquet文件: {file_path}")
        return pd.read_parquet(file_path)

    csv_files = [f for f in files if f.endswith('.csv')]
    if csv_files:
        file_path = os.path.join(raw_dir, csv_files[0])
        print(f"加载csv文件: {file_path}")
        return pd.read_csv(file_path)

    raise FileNotFoundError(f"在{raw_dir}中未找到parquet或csv文件")

df = load_local_dataset(RAW_DATA_DIR)
print(f"原始数据规模：{len(df)}条")
print("数据样例：")
print(df.head(2))

# 3. 列名检查与映射
required_columns = ['review', 'label']
if not set(required_columns).issubset(df.columns):
    print(f"数据列名: {df.columns.tolist()}")
    column_map = {}
    if 'comment' in df.columns:
        column_map['comment'] = 'review'
    if 'sentiment' in df.columns:
        column_map['sentiment'] = 'label'
    df = df.rename(columns=column_map)

# 4. 数据平衡处理：上采样 Positive 类
positive_df = df[df['label'] == 1]
negative_df = df[df['label'] == 0]

if len(positive_df) < len(negative_df):
    positive_df = resample(
        positive_df,
        replace=True,  # 有放回采样
        n_samples=len(negative_df),  # 采样到和 Negative 一样多
        random_state=42
    )
    df = pd.concat([positive_df, negative_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"上采样后数据规模：{len(df)}条（Positive 与 Negative 数量平衡）")

# 5. 构造SFT格式
def format_sft_data(row):
    instruction = "分析用户对美团外卖的评论情感，并为商家提供1条改进建议"
    input_text = row['review']

    if row['label'] == 1:
        sentiment = "积极"
        suggestion = "保持当前服务质量，可适当推出老客优惠活动提升复购率"
    else:
        sentiment = "消极"
        if "慢" in input_text or "久" in input_text:
            suggestion = "优化配送路线规划，高峰期提前调配骑手"
        elif "差" in input_text or "不好吃" in input_text:
            suggestion = "检查食材新鲜度，加强厨师培训"
        else:
            suggestion = "收集更多用户反馈，针对性优化服务"

    output_text = f"情感倾向：{sentiment}\n改进建议：{suggestion}"
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }

sft_data = df.apply(format_sft_data, axis=1).tolist()
train_data, val_data = train_test_split(sft_data, test_size=0.2, random_state=42)

# 6. 保存数据
with open(f"{DATA_SAVE_PATH}/train.jsonl", "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
with open(f"{DATA_SAVE_PATH}/val.jsonl", "w", encoding="utf-8") as f:
    for item in val_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"训练集：{len(train_data)}条（保存至{DATA_SAVE_PATH}/train.jsonl）")
print(f"验证集：{len(val_data)}条（保存至{DATA_SAVE_PATH}/val.jsonl）")