import os
import json
import torch
import pandas as pd
import numpy as np
import jieba
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import swanlab


VERSION = "1.0"
BASE_MODEL_PATH = "/root/autodl-tmp/Qwen3-1.7B"
DATA_PATH = "/root/autodl-tmp/meituan_comment/data/raw/meituan.csv"
OUTPUT_DIR = f"/root/autodl-tmp/meituan_comment/output_model/output_v{VERSION}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 过滤numpy警告
np.warnings.filterwarnings('ignore', category=FutureWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# ====================
# 2. 加载分词器（保持与评估一致）
# ====================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    padding_side="right"
)
tokenizer.pad_token = tokenizer.eos_token

# ====================
# 3. 加载模型（结构不变，保留情感识别能力）
# ====================
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# ====================
# 4. LoRA配置（不变，因情感识别已达标）
# ====================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ====================
# 5. 数据加载与预处理（核心优化：新增核心词检查，过滤无效回复）
# ====================
def extract_core_words(text):
    """提取文本中的核心词（2字以上名词/形容词，适配餐饮场景）"""
    if not text:
        return set()
    # 过滤无意义词，保留与餐饮相关的核心词
    stop_words = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "个", "也", "很"}
    words = set([w for w in jieba.lcut(text) if len(w) >= 2 and w not in stop_words])
    # 补充餐饮领域核心词表，提升提取准确性
    food_core_words = {"菜", "饭", "面", "汤", "肉", "鱼", "蛋", "虾", "咸", "淡", "辣", "凉", "热", "冷", "硬", "软", "多", "少", "足", "够", "快", "慢", "好", "差", "优", "劣", "鲜", "腥", "臭", "香", "贵", "便宜", "值", "不值", "包装", "分量", "味道", "口感", "服务", "态度", "配送", "时效", "保温", "卫生"}
    return words & food_core_words  # 只保留餐饮相关核心词

def load_and_split_data(path, test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    df = df.dropna(subset=['label', 'review', 'repeat'])
    
    # 新增：核心过滤条件——回复必须包含评论的至少1个核心词
    core_word_filtered = []
    for _, row in df.iterrows():
        review = row['review'].strip()
        response = row['repeat'].strip()
        # 提取评论核心词
        review_core_words = extract_core_words(review)
        # 检查回复是否包含至少1个核心词（不区分大小写）
        if review_core_words and any(word in response.lower() for word in review_core_words):
            core_word_filtered.append(row)
    
    # 基于核心词过滤后的结果继续清洗
    df = pd.DataFrame(core_word_filtered)
    df = df[
        (df['label'].isin([0, 1])) & 
        (df['review'].str.len().between(10, 200)) & 
        (df['repeat'].str.len().between(15, 100))
    ]
    
    # 难例标记（保留原有逻辑，提升模型鲁棒性）
    df['is_hard'] = 0
    df.loc[df['review'].str.contains('但|可是|不过|虽然|但是'), 'is_hard'] = 1
    df.loc[(df['label'] == 0) & df['review'].str.contains('好|不错|满意|快|足'), 'is_hard'] = 1
    
    # 分层抽样（保持数据分布）
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['label']
    )
    print(f"数据加载完成：训练集{len(train_df)}条，验证集{len(val_df)}条（均满足'回复含评论核心词'）")
    return Dataset.from_pandas(train_df), Dataset.from_pandas(val_df)

train_dataset, val_dataset = load_and_split_data(DATA_PATH)

# ====================
# 6. Prompt格式化（核心优化：强化JSON格式约束+核心词要求）
# ====================
def format_prompt(sample):
    sentiment = "积极" if sample['label'] == 1 else "消极"
    review_clean = sample['review'].replace('"', '\\"').replace('\n', '').strip()  # 转义双引号，避免JSON语法错误
    response_clean = sample['repeat'].replace('"', '\\"').replace('\n', '').strip()
    
    # 双重验证：确保回复包含评论核心词（避免训练数据漏检）
    review_core_words = extract_core_words(review_clean)
    assert any(word in response_clean for word in review_core_words), f"回复缺失核心词：评论{review_clean} | 回复{response_clean}"
    
    # 生成严格JSON（用indent=0压缩格式，避免模型学习多余换行）
    structured_output = json.dumps({
        "sentiment": sentiment,          # 严格值："积极"或"消极"
        "key_issue": review_clean,       # 严格保留评论核心内容
        "response_to_customer": response_clean  # 严格包含评论核心词
    }, ensure_ascii=False, indent=0)
    
    # 强化system指令：用"必须"明确约束，优先级排序（格式>核心词>回复质量）
    return f"""<system>你必须严格遵守以下规则，否则输出无效：
1. 输出内容：仅包含JSON，无任何多余文本（包括注释、空格、换行）；
2. JSON格式要求：
   - 键名严格为"sentiment"、"key_issue"、"response_to_customer"（无其他键）；
   - "sentiment"的值仅能是"积极"或"消极"（无其他值）；
   - 字符串用双引号包裹，逗号后无空格，括号完全闭合；
3. 内容要求：
   - "response_to_customer"必须包含评论中的至少1个核心词（如评论提"菜咸"，回复必须含"咸"）；
   - "key_issue"必须提取评论的核心矛盾或优点（不超过50字）；
4. 错误处理：若无法满足上述任何一条，输出{{"error":"invalid_output"}}。</system>
<user>用户评论: {review_clean}</user>
<assistant>{structured_output}</assistant>"""

# ====================
# 7. Tokenize函数（不变，保留原有逻辑）
# ====================
def tokenize_function(examples):
    prompt = format_prompt(examples)
    
    inputs = tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    input_ids = inputs["input_ids"][0].tolist()
    attention_mask = inputs["attention_mask"][0].tolist()
    
    # 定位<assistant>位置，只对输出部分计算损失
    assistant_token = tokenizer("<assistant>", add_special_tokens=False)["input_ids"]
    assistant_start = None
    for i in range(len(input_ids) - len(assistant_token) + 1):
        if input_ids[i:i+len(assistant_token)] == assistant_token:
            assistant_start = i + len(assistant_token)
            break
    
    # 构建labels：仅训练<assistant>后的输出部分
    labels = [-100] * len(input_ids)
    if assistant_start is not None:
        for i in range(assistant_start, len(input_ids)):
            if input_ids[i] != tokenizer.pad_token_id:
                labels[i] = input_ids[i]
            else:
                labels[i] = -100
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sample_weight": 2.0 if examples.get('is_hard', 0) == 1 else 1.0  # 难例权重翻倍
    }

# 应用tokenize（保持逐条处理，确保格式定位准确）
tokenized_train = train_dataset.map(
    tokenize_function, 
    remove_columns=train_dataset.column_names,
    batched=False,
    keep_in_memory=True
)
tokenized_val = val_dataset.map(
    tokenize_function, 
    remove_columns=val_dataset.column_names,
    batched=False,
    keep_in_memory=True
)

# 显式指定保留的列，避免样本权重丢失
tokenized_train = tokenized_train.with_format("torch", columns=["input_ids", "attention_mask", "labels", "sample_weight"])
tokenized_val = tokenized_val.with_format("torch", columns=["input_ids", "attention_mask", "labels", "sample_weight"])

# ====================
# 8. 自定义数据处理器（不变，保留样本权重逻辑）
# ====================
class WeightedDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features, return_tensors=None):
        try:
            sample_weights = [f["sample_weight"] for f in features]
            for f in features:
                if "sample_weight" in f:
                    del f["sample_weight"]
        except KeyError:
            sample_weights = [1.0 for _ in features]
        
        batch = super().__call__(features, return_tensors=return_tensors)
        batch["sample_weights"] = torch.tensor(sample_weights, dtype=torch.float32, device=batch["input_ids"].device)
        return batch

data_collator = WeightedDataCollator(tokenizer=tokenizer, mlm=False)

# ====================
# 9. 训练参数（微调：降低学习率，避免过拟合）
# ====================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=8,  # 减少2个epoch，避免格式过拟合
    learning_rate=1.5e-5,  # 降低学习率，让模型更慢地学习格式规则
    logging_steps=20,
    save_total_limit=2,
    warmup_steps=150,  # 减少warmup步数，加快格式学习
    weight_decay=0.01,
    fp16=True,
    report_to="swanlab",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False
)

# ====================
# 10. 兼容接口的自定义Trainer（不变）
# ====================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weights = inputs.pop("sample_weights", None)
        
        outputs = model(**inputs)
        loss = outputs.loss
        
        if sample_weights is not None:
            weighted_loss = (loss * sample_weights).mean()
        else:
            weighted_loss = loss
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss

# ====================
# 11. 初始化Trainer并训练
# ====================
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator
)

# 用新版本名初始化swanlab，便于对比实验
swanlab.init(project="meituan-qwen-finetune", experiment_name=f"lora-tuning-v{VERSION}")
trainer.train()

# ====================
# 12. 保存模型（新增：保存核心词提取函数，便于评估时复用）
# ====================
model.save_pretrained(f"{OUTPUT_DIR}/lora_weights")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/tokenizer")

# 保存核心词提取函数，评估时直接调用（避免代码冗余）
core_word_code = """import jieba

def extract_core_words(text):
    stop_words = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "个", "也", "很"}
    words = set([w for w in jieba.lcut(text) if len(w) >= 2 and w not in stop_words])
    food_core_words = {"菜", "饭", "面", "汤", "肉", "鱼", "蛋", "虾", "咸", "淡", "辣", "凉", "热", "冷", "硬", "软", "多", "少", "足", "够", "快", "慢", "好", "差", "优", "劣", "鲜", "腥", "臭", "香", "贵", "便宜", "值", "不值", "包装", "分量", "味道", "口感", "服务", "态度", "配送", "时效", "保温", "卫生"}
    return words & food_core_words
"""
with open(f"{OUTPUT_DIR}/core_word_utils.py", "w", encoding="utf-8") as f:
    f.write(core_word_code)

print(f"模型保存至：{OUTPUT_DIR}")
print(f"核心词提取工具保存至：{OUTPUT_DIR}/core_word_utils.py")