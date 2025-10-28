import os
import json
import torch
import pandas as pd
import numpy as np
import jieba
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# ====================
# 1. 基础配置（适配v1.0模型，与训练路径对齐）
# ====================
VERSION = "1.0"
BASE_MODEL_PATH = "/root/autodl-tmp/Qwen3-1.7B"
# 指向v6.0模型的LoRA权重
LORA_PATH = f"/root/autodl-tmp/meituan_comment/output_model/output_v{VERSION}/lora_weights"
TOKENIZER_PATH = f"/root/autodl-tmp/meituan_comment/output_model/output_v{VERSION}/tokenizer"
DATA_PATH = "/root/autodl-tmp/meituan_comment/data/raw/meituan.csv"
# 验证结果保存路径（与训练输出区分开）
OUTPUT_DIR = f"/root/autodl-tmp/meituan/output_eval/evaluation_results_v{VERSION}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 4090批量推理优化参数
BATCH_SIZE = 16  # 24G显存稳定运行，OOM时降为8
MAX_PROMPT_LENGTH = 512  # 与训练一致
MAX_NEW_TOKENS = 120  # 适当增加，适配v6.0更严格的JSON格式
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ====================
# 2. 导入训练时的核心词提取函数（确保逻辑完全一致）
# ====================
def extract_core_words(text):
    """复用训练时的核心词提取逻辑，避免评估标准偏差"""
    if not text:
        return set()
    stop_words = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "个", "也", "很"}
    words = set([w for w in jieba.lcut(text) if len(w) >= 2 and w not in stop_words])
    # 完全复用训练时的餐饮领域核心词表
    food_core_words = {"菜", "饭", "面", "汤", "肉", "鱼", "蛋", "虾", "咸", "淡", "辣", "凉", "热", "冷", "硬", "软", "多", "少", "足", "够", "快", "慢", "好", "差", "优", "劣", "鲜", "腥", "臭", "香", "贵", "便宜", "值", "不值", "包装", "分量", "味道", "口感", "服务", "态度", "配送", "时效", "保温", "卫生"}
    return words & food_core_words

# ====================
# 3. LoRA文件检查（支持safetensors/bin格式）
# ====================
def check_lora_files(lora_path):
    required_files = ["adapter_config.json"]
    weight_files = ["adapter_model.safetensors", "adapter_model.bin"]
    # 检查配置文件
    for file in required_files:
        if not os.path.exists(os.path.join(lora_path, file)):
            raise FileNotFoundError(f"LoRA配置文件缺失：{os.path.join(lora_path, file)}")
    # 检查权重文件（二选一）
    if not any(os.path.exists(os.path.join(lora_path, wf)) for wf in weight_files):
        raise FileNotFoundError(f"LoRA权重文件缺失，需存在以下之一：{weight_files}")
    return True

# 执行文件检查
check_lora_files(LORA_PATH)
print(f"=== 验证配置（v{VERSION}） ===\nDevice: {DEVICE}\nBatch Size: {BATCH_SIZE}\nLoRA Path: {LORA_PATH}\nOutput Path: {OUTPUT_DIR}\n")

# ====================
# 4. 加载验证集（复用训练的数据清洗逻辑）
# ====================
def load_validation_data(path, test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    df = df.dropna(subset=['label', 'review', 'repeat'])
    
    # 复用训练时的核心词过滤逻辑（仅用于验证集划分，确保数据分布一致）
    core_word_filtered = []
    for _, row in df.iterrows():
        review = row['review'].strip()
        response = row['repeat'].strip()
        review_core = extract_core_words(review)
        if review_core and any(word in response for word in review_core):
            core_word_filtered.append(row)
    
    df = pd.DataFrame(core_word_filtered)
    df = df[
        (df['label'].isin([0, 1])) & 
        (df['review'].str.len().between(10, 200)) & 
        (df['repeat'].str.len().between(15, 100))
    ]
    
    # 与训练完全一致的分层抽样
    _, val_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['label']
    )
    print(f"验证集加载完成：{len(val_df)}条样本 | 消极占比：{val_df['label'].value_counts(normalize=True)[0]:.2%} | 积极占比：{val_df['label'].value_counts(normalize=True)[1]:.2%}")
    return val_df

val_df = load_validation_data(DATA_PATH)

# ====================
# 5. 加载v6.0模型与分词器（适配严格格式生成）
# ====================
def load_lora_model():
    # 加载训练时保存的分词器（避免格式不兼容）
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载基础模型 + LoRA权重
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 4090 FP16推理加速
        device_map=DEVICE,
        low_cpu_mem_usage=True
    )
    
    # 加载v6.0的LoRA权重（自动支持safetensors）
    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    
    # 合并权重提升推理速度（4090显存足够）
    model = model.merge_and_unload()
    model.eval()  # 禁用Dropout，确保生成稳定性
    print(f"v{VERSION}模型加载完成（LoRA权重已合并）")
    return model, tokenizer

model, tokenizer = load_lora_model()

# ====================
# 6. 批量推理函数（适配v6.0的Prompt格式）
# ====================
def batch_generate_reviews(reviews):
    prompts = []
    for review in reviews:
        # 与训练时完全一致的文本清洗
        review_clean = review.replace('"', '\\"').replace('\n', '').strip()
        # 复用训练时的system指令（确保模型接收相同约束）
        prompt = f"""<system>你必须严格遵守以下规则，否则输出无效：
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
<assistant>"""
        prompts.append(prompt)
    
    # 批量分词（统一长度，减少显存波动）
    inputs = tokenizer(
        prompts,
        max_length=MAX_PROMPT_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(DEVICE)
    
    # 批量生成（降低随机性，适配v6.0的严格格式要求）
    with torch.no_grad():  # 禁用梯度，节省显存
        with torch.cuda.amp.autocast(dtype=torch.float16):  # 强制FP16加速
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.4,  # 降低随机性（v6.0关键：从0.7→0.4）
                top_p=0.85,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1
            )
    
    # 解码生成结果（仅保留<assistant>后的内容）
    generated_texts = []
    for i in range(len(outputs)):
        generated = tokenizer.decode(
            outputs[i][len(inputs["input_ids"][i]):],
            skip_special_tokens=True
        )
        generated_texts.append(generated)
    
    return generated_texts

# ====================
# 7. 结果解析函数（适配v6.0的JSON格式与错误处理）
# ====================
def parse_generated_result(generated_text, review):
    """增强解析逻辑，适配v6.0的"error"输出和核心词检查"""
    try:
        # 处理v6.0可能的错误输出
        if '{"error":"invalid_output"}' in generated_text:
            return {
                "sentiment_pred": -1,
                "key_issue_pred": "",
                "response_pred": "",
                "parse_status": "model_error: invalid_output",
                "core_coverage": 0.0
            }
        
        # 提取JSON片段（处理可能的多余字符）
        if '{' in generated_text and '}' in generated_text:
            json_start = generated_text.index('{')
            json_end = generated_text.rindex('}') + 1
            json_str = generated_text[json_start:json_end]
            
            # 解析JSON（处理转义字符）
            result = json.loads(json_str)
            
            # 验证键名完整性（v6.0要求的3个键）
            required_keys = {"sentiment", "key_issue", "response_to_customer"}
            if not required_keys.issubset(result.keys()):
                missing_keys = required_keys - result.keys()
                return {
                    "sentiment_pred": -1,
                    "key_issue_pred": "",
                    "response_pred": "",
                    "parse_status": f"missing_keys: {missing_keys}",
                    "core_coverage": 0.0
                }
            
            # 映射情感标签（严格匹配"积极"/"消极"）
            sentiment = result["sentiment"].strip()
            if sentiment not in ["积极", "消极"]:
                return {
                    "sentiment_pred": -1,
                    "key_issue_pred": result["key_issue"].strip(),
                    "response_pred": result["response_to_customer"].strip(),
                    "parse_status": f"invalid_sentiment: {sentiment}",
                    "core_coverage": 0.0
                }
            sentiment_pred = 1 if sentiment == "积极" else 0
            
            # 计算核心词覆盖率（复用训练时的函数）
            review_core = extract_core_words(review)
            response = result["response_to_customer"].strip()
            response_core = extract_core_words(response)
            core_coverage = len(review_core & response_core) / len(review_core) if review_core else 0.0
            
            return {
                "sentiment_pred": sentiment_pred,
                "key_issue_pred": result["key_issue"].strip(),
                "response_pred": response,
                "parse_status": "success",
                "core_coverage": core_coverage
            }
        else:
            return {
                "sentiment_pred": -1,
                "key_issue_pred": "",
                "response_pred": generated_text.strip()[:50] + "...",
                "parse_status": "no_json_structure",
                "core_coverage": 0.0
            }
    except json.JSONDecodeError as e:
        return {
            "sentiment_pred": -1,
            "key_issue_pred": "",
            "response_pred": generated_text.strip()[:50] + "...",
            "parse_status": f"json_error: {str(e)[:20]}",
            "core_coverage": 0.0
        }
    except Exception as e:
        return {
            "sentiment_pred": -1,
            "key_issue_pred": "",
            "response_pred": generated_text.strip()[:50] + "...",
            "parse_status": f"other_error: {str(e)[:20]}",
            "core_coverage": 0.0
        }

# ====================
# 8. 核心验证指标计算（适配v6.0优化目标）
# ====================
def calculate_metrics(results_df):
    # 基础统计
    total_samples = len(results_df)
    success_samples = results_df[results_df["parse_status"] == "success"]
    parse_success_rate = len(success_samples) / total_samples if total_samples > 0 else 0
    
    # 8.1 情感识别指标（仅用成功解析的样本）
    sentiment_metrics = {
        "accuracy": 0, "f1_weighted": 0, "f1_macro": 0,
        "confusion_matrix": np.zeros((2, 2)), "classification_report": {}
    }
    if len(success_samples) > 0:
        y_true = success_samples["true_label"]
        y_pred = success_samples["sentiment_pred"]
        sentiment_metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(
                y_true, y_pred, target_names=["Negative(0)", "Positive(1)"], output_dict=True
            )
        }
    
    # 8.2 回复质量指标（v6.0核心优化目标）
    response_metrics = {
        "avg_core_coverage": success_samples["core_coverage"].mean() if len(success_samples) > 0 else 0,
        "high_coverage_rate": len(success_samples[success_samples["core_coverage"] >= 0.5]) / len(success_samples) if len(success_samples) > 0 else 0,
        "avg_response_length": success_samples["response_pred"].str.len().mean() if len(success_samples) > 0 else 0,
        "valid_response_rate": len(success_samples[success_samples["response_pred"].str.len() >= 10]) / len(success_samples) if len(success_samples) > 0 else 0
    }
    
    # 8.3 格式错误分析（v6.0新增：定位格式问题类型）
    error_analysis = results_df["parse_status"].value_counts().to_dict()
    
    # 8.4 关键信息提取指标（ROUGE-L，适配中文）
    def calculate_rouge_l(reference, hypothesis):
        if not reference or not hypothesis:
            return 0.0
        ref_words = jieba.lcut(reference)
        hyp_words = jieba.lcut(hypothesis)
        # 计算最长公共子序列
        lcs_mat = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    lcs_mat[i][j] = lcs_mat[i-1][j-1] + 1
                else:
                    lcs_mat[i][j] = max(lcs_mat[i-1][j], lcs_mat[i][j-1])
        lcs_len = lcs_mat[-1][-1]
        if lcs_len == 0:
            return 0.0
        precision = lcs_len / len(hyp_words)
        recall = lcs_len / len(ref_words)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    success_samples["rouge_l"] = success_samples.apply(
        lambda x: calculate_rouge_l(x["true_review"], x["key_issue_pred"]), axis=1
    )
    key_issue_metrics = {
        "avg_rouge_l": success_samples["rouge_l"].mean() if len(success_samples) > 0 else 0
    }
    
    return {
        "parse_success_rate": parse_success_rate,
        "sentiment_metrics": sentiment_metrics,
        "response_metrics": response_metrics,
        "key_issue_metrics": key_issue_metrics,
        "error_analysis": error_analysis,
        "success_samples": success_samples,
        "total_samples": total_samples
    }

# ====================
# 9. 可视化函数（英文标签，无中文字体依赖）
# ====================
def plot_evaluation_results(metrics, save_dir):
    # 9.1 混淆矩阵（情感识别结果）
    conf_matrix = metrics["sentiment_metrics"]["confusion_matrix"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative(0)", "Positive(1)"],
        yticklabels=["Negative(0)", "Positive(1)"],
        cbar_kws={"label": "Sample Count"}
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(f"Sentiment Classification Confusion Matrix (v{VERSION})\nParsing Success Rate: {metrics['parse_success_rate']:.2%}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 9.2 核心指标柱状图（v6.0优化目标重点展示）
    metrics_data = {
        "Sentiment Accuracy": metrics["sentiment_metrics"]["accuracy"],
        "JSON Parsing Rate": metrics["parse_success_rate"],
        "Avg. Core Coverage": metrics["response_metrics"]["avg_core_coverage"],
        "High Coverage Rate (≥50%)": metrics["response_metrics"]["high_coverage_rate"],
        "Key Issue ROUGE-L": metrics["key_issue_metrics"]["avg_rouge_l"]
    }
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        metrics_data.keys(),
        metrics_data.values(),
        color=["#2E86AB", "#E74C3C", "#27AE60", "#F39C12", "#8E44AD"]
    )
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10
        )
    plt.ylim(0, 1.1)
    plt.ylabel("Metric Value (Higher = Better)", fontsize=12)
    plt.title(f"Core Evaluation Metrics (v{VERSION})", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/core_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 9.3 格式错误分布（v6.0新增：定位格式问题）
    error_data = metrics["error_analysis"]
    if len(error_data) > 1 or (len(error_data) == 1 and "success" not in error_data):
        plt.figure(figsize=(10, 6))
        # 过滤success，只展示错误类型
        error_only = {k: v for k, v in error_data.items() if k != "success"}
        if error_only:
            plt.pie(
                error_only.values(),
                labels=error_only.keys(),
                autopct="%1.1f%%",
                startangle=90,
                colors=sns.color_palette("Set3")
            )
            plt.title(f"Format Error Distribution (v{VERSION})\nTotal Error Samples: {sum(error_only.values())}", fontsize=14)
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/error_distribution.png", dpi=300, bbox_inches="tight")
            plt.close()
    
    print(f"可视化图表已保存至：{save_dir}")

# ====================
# 10. 主验证流程
# ====================
def main_evaluation():
    # 批量推理（分批次处理，避免显存溢出）
    results = []
    total_batches = (len(val_df) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in tqdm(range(total_batches), desc=f"v{VERSION} Batch Inference"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(val_df))
        batch_data = val_df.iloc[start_idx:end_idx]
        
        # 批量生成
        reviews = batch_data["review"].tolist()
        generated_texts = batch_generate_reviews(reviews)
        
        # 解析结果并保存
        for idx in range(len(batch_data)):
            row = batch_data.iloc[idx]
            parsed = parse_generated_result(generated_texts[idx], row["review"])
            results.append({
                "true_label": row["label"],
                "true_review": row["review"].strip(),
                "true_response": row["repeat"].strip(),
                "sentiment_pred": parsed["sentiment_pred"],
                "key_issue_pred": parsed["key_issue_pred"],
                "response_pred": parsed["response_pred"],
                "parse_status": parsed["parse_status"],
                "core_coverage": parsed["core_coverage"]
            })
    
    # 保存原始结果（含错误样本，便于问题定位）
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        f"{OUTPUT_DIR}/raw_evaluation_results.csv",
        index=False,
        encoding="utf-8"
    )
    
    # 计算验证指标
    metrics = calculate_metrics(results_df)
    
    # 保存成功样本结果
    success_df = metrics["success_samples"]
    success_df.to_csv(
        f"{OUTPUT_DIR}/success_evaluation_results.csv",
        index=False,
        encoding="utf-8"
    )
    
    # 生成英文验证报告（便于无中文字体查看）
    report = f"""# Meituan Review Model Validation Report (v{VERSION})

## 1. Basic Information
- Total Validation Samples: {metrics['total_samples']}
- Successfully Parsed Samples: {len(metrics['success_samples'])}
- JSON Parsing Success Rate: {metrics['parse_success_rate']:.2%}
- Inference Device: {DEVICE} | Batch Size: {BATCH_SIZE}
- Validation Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 2. Sentiment Classification (Main Task)
| Metric               | Value       |
|----------------------|-------------|
| Accuracy             | {metrics['sentiment_metrics']['accuracy']:.4f} |
| Weighted F1 Score    | {metrics['sentiment_metrics']['f1_weighted']:.4f} |
| Macro F1 Score       | {metrics['sentiment_metrics']['f1_macro']:.4f} |

### Classification Details
- Negative Recall: {metrics['sentiment_metrics']['classification_report'].get('Negative(0)', {}).get('recall', 0):.4f}
- Positive Recall: {metrics['sentiment_metrics']['classification_report'].get('Positive(1)', {}).get('recall', 0):.4f}
- Negative Precision: {metrics['sentiment_metrics']['classification_report'].get('Negative(0)', {}).get('precision', 0):.4f}
- Positive Precision: {metrics['sentiment_metrics']['classification_report'].get('Positive(1)', {}).get('precision', 0):.4f}

## 3. Response Quality (v{VERSION} Key Optimization Target)
| Metric                     | Value                 |
|----------------------------|-----------------------|
| Avg. Core Word Coverage    | {metrics['response_metrics']['avg_core_coverage']:.4f} |
| High Coverage Rate (≥50%)  | {metrics['response_metrics']['high_coverage_rate']:.2%} |
| Avg. Response Length       | {metrics['response_metrics']['avg_response_length']:.1f} chars |
| Valid Response Rate (≥10 chars) | {metrics['response_metrics']['valid_response_rate']:.2%} |

## 4. Key Issue Extraction
- Avg. ROUGE-L: {metrics['key_issue_metrics']['avg_rouge_l']:.4f}

## 5. Format Error Analysis
| Error Type               | Sample Count | Percentage |
"""
    # 添加错误分析表格
    total_errors = metrics['total_samples'] - len(metrics['success_samples'])
    for error_type, count in metrics['error_analysis'].items():
        if error_type == "success":
            continue
        percentage = (count / total_errors) * 100 if total_errors > 0 else 0
        report += f"| {error_type:<20} | {count:<11} | {percentage:<10.1f}%\n"
    
    # 添加混淆矩阵
    conf_matrix = metrics['sentiment_metrics']['confusion_matrix']
    report += f"""
## 6. Confusion Matrix
| True \\ Predicted | Negative(0) | Positive(1) | Total |
|-------------------|-------------|-------------|-------|
| Negative(0)       | {conf_matrix[0][0]} | {conf_matrix[0][1]} | {conf_matrix[0].sum()} |
| Positive(1)       | {conf_matrix[1][0]} | {conf_matrix[1][1]} | {conf_matrix[1].sum()} |
| Total             | {conf_matrix[:,0].sum()} | {conf_matrix[:,1].sum()} | {conf_matrix.sum()} |

## 7. Validation Conclusion
- JSON Parsing: {'✅ Pass' if metrics['parse_success_rate'] >= 0.9 else '❌ Fail'} (Target: ≥90%)
- Core Word Coverage: {'✅ Pass' if metrics['response_metrics']['avg_core_coverage'] >= 0.5 else '❌ Fail'} (Target: ≥50%)
- Sentiment Accuracy: {'✅ Pass' if metrics['sentiment_metrics']['accuracy'] >= 0.95 else '❌ Fail'} (Target: ≥95%)
"""
    
    # 保存报告
    with open(f"{OUTPUT_DIR}/validation_report_en.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    # 生成可视化
    plot_evaluation_results(metrics, OUTPUT_DIR)
    
    # 打印关键结果
    print("="*80)
    print(f"v{VERSION} Model Validation Completed!")
    print(f"[Key Results] Parsing Rate: {metrics['parse_success_rate']:.2%} | Core Coverage: {metrics['response_metrics']['avg_core_coverage']:.4f} | Sentiment Acc: {metrics['sentiment_metrics']['accuracy']:.4f}")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("="*80)

# 启动验证
if __name__ == "__main__":
    main_evaluation()