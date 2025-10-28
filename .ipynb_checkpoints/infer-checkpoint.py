import os
import json
import torch
import pandas as pd
import jieba
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ====================
# 1. 基础配置（与v1.0训练/验证一致）
# ====================
VERSION = "1.0"
BASE_MODEL_PATH = "/root/autodl-tmp/Qwen3-1.7B"
LORA_PATH = f"/root/autodl-tmp/meituan_comment/output_model/output_v{VERSION}/lora_weights"
TOKENIZER_PATH = f"/root/autodl-tmp/meituan_comment/output_model/output_v{VERSION}/tokenizer"

# 4090推理优化参数
BATCH_SIZE = 16  # 24G显存稳定，OOM时降为8
MAX_PROMPT_LENGTH = 512
MAX_NEW_TOKENS = 120  # 适配v6.0严格JSON格式
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"=== 美团评论推理配置（v{VERSION}） ===\n设备：{DEVICE}\n批量大小：{BATCH_SIZE}\n")

# ====================
# 2. 核心工具函数（复用训练逻辑）
# ====================
def extract_core_words(text):
    """复用训练时的核心词提取逻辑，用于结果校验"""
    if not text:
        return set()
    stop_words = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "个", "也", "很"}
    words = set([w for w in jieba.lcut(text) if len(w) >= 2 and w not in stop_words])
    food_core_words = {"菜", "饭", "面", "汤", "肉", "鱼", "蛋", "虾", "咸", "淡", "辣", "凉", "热", "冷", "硬", "软", "多", "少", "足", "够", "快", "慢", "好", "差", "优", "劣", "鲜", "腥", "臭", "香", "贵", "便宜", "值", "不值", "包装", "分量", "味道", "口感", "服务", "态度", "配送", "时效", "保温", "卫生"}
    return words & food_core_words

def format_inference_prompt(review):
    """生成与训练一致的推理Prompt，确保格式兼容"""
    review_clean = review.replace('"', '\\"').replace('\n', '').strip()
    return f"""<system>你必须严格遵守以下规则，否则输出无效：
1. 输出内容：仅包含JSON，无任何多余文本（包括注释、空格、换行）；
2. JSON格式要求：
   - 键名严格为"sentiment"、"key_issue"、"response_to_customer"（无其他键）；
   - "sentiment"的值仅能是"积极"或"消极"（无其他值）；
   - 字符串用双引号包裹，逗号后无空格，括号完全闭合；
3. 内容要求：
   - "response_to_customer"必须包含评论中的至少1个核心词；
   - "key_issue"提取评论核心（不超过50字）；
4. 错误处理：若无法满足，输出{{"error":"invalid_output"}}。</system>
<user>用户评论: {review_clean}</user>
<assistant>"""

# ====================
# 3. 加载v6.0模型与分词器（推理优化）
# ====================
def load_inference_model():
    """加载合并LoRA权重的推理模型，提升速度"""
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 4090 FP16推理加速
        device_map=DEVICE,
        low_cpu_mem_usage=True
    )
    
    # 加载并合并LoRA权重（推理时合并，速度提升30%+）
    model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    model = model.merge_and_unload()
    model.eval()  # 禁用Dropout，确保推理稳定
    
    # 推理优化：启用FlashAttention（若支持）
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    print(f"v{VERSION}推理模型加载完成（LoRA权重已合并）")
    return model, tokenizer

# 加载模型（全局唯一，避免重复加载）
model, tokenizer = load_inference_model()

# ====================
# 4. 推理核心函数（支持单条/批量）
# ====================
def parse_inference_result(generated_text, review):
    """解析模型输出，返回结构化结果+质量校验"""
    try:
        # 处理模型错误输出
        if '{"error":"invalid_output"}' in generated_text:
            return {
                "review": review,
                "sentiment": None,
                "sentiment_label": -1,  # -1=未知，0=消极，1=积极
                "key_issue": None,
                "response_to_customer": None,
                "core_coverage": 0.0,
                "status": "failed",
                "error_msg": "模型无法生成有效输出"
            }
        
        # 提取JSON片段（兼容多余字符）
        if '{' in generated_text and '}' in generated_text:
            json_start = generated_text.index('{')
            json_end = generated_text.rindex('}') + 1
            json_str = generated_text[json_start:json_end]
            result = json.loads(json_str)
            
            # 校验键名完整性
            required_keys = {"sentiment", "key_issue", "response_to_customer"}
            if not required_keys.issubset(result.keys()):
                missing = required_keys - result.keys()
                return {
                    "review": review,
                    "sentiment": None,
                    "sentiment_label": -1,
                    "key_issue": None,
                    "response_to_customer": None,
                    "core_coverage": 0.0,
                    "status": "failed",
                    "error_msg": f"缺失键名：{missing}"
                }
            
            # 校验情感标签
            sentiment = result["sentiment"].strip()
            if sentiment not in ["积极", "消极"]:
                return {
                    "review": review,
                    "sentiment": sentiment,
                    "sentiment_label": -1,
                    "key_issue": result["key_issue"].strip(),
                    "response_to_customer": result["response_to_customer"].strip(),
                    "core_coverage": 0.0,
                    "status": "failed",
                    "error_msg": f"情感标签无效（需为'积极'/'消极'）"
                }
            
            # 计算核心词覆盖率（质量校验）
            review_core = extract_core_words(review)
            response = result["response_to_customer"].strip()
            response_core = extract_core_words(response)
            core_coverage = len(review_core & response_core) / len(review_core) if review_core else 0.0
            
            # 返回成功结果
            return {
                "review": review,
                "sentiment": sentiment,
                "sentiment_label": 1 if sentiment == "积极" else 0,
                "key_issue": result["key_issue"].strip(),
                "response_to_customer": response,
                "core_coverage": round(core_coverage, 4),
                "status": "success",
                "error_msg": None
            }
        
        # 无JSON结构的情况
        return {
            "review": review,
            "sentiment": None,
            "sentiment_label": -1,
            "key_issue": None,
            "response_to_customer": None,
            "core_coverage": 0.0,
            "status": "failed",
            "error_msg": "输出无JSON结构"
        }
    
    except json.JSONDecodeError as e:
        return {
            "review": review,
            "sentiment": None,
            "sentiment_label": -1,
            "key_issue": None,
            "response_to_customer": None,
            "core_coverage": 0.0,
            "status": "failed",
            "error_msg": f"JSON解析错误：{str(e)[:30]}"
        }
    except Exception as e:
        return {
            "review": review,
            "sentiment": None,
            "sentiment_label": -1,
            "key_issue": None,
            "response_to_customer": None,
            "core_coverage": 0.0,
            "status": "failed",
            "error_msg": f"其他错误：{str(e)[:30]}"
        }

def batch_inference(reviews):
    """批量推理函数（4090优化）"""
    if not reviews:
        return []
    
    # 分批次处理（避免显存溢出）
    results = []
    total_batches = (len(reviews) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in tqdm(range(total_batches), desc="批量推理进度"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, len(reviews))
        batch_reviews = reviews[start_idx:end_idx]
        
        # 1. 构建批量Prompt
        prompts = [format_inference_prompt(review) for review in batch_reviews]
        
        # 2. 批量分词
        inputs = tokenizer(
            prompts,
            max_length=MAX_PROMPT_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(DEVICE)
        
        # 3. 批量生成（低随机性，确保格式稳定）
        with torch.no_grad():  # 禁用梯度，节省显存
            with torch.cuda.amp.autocast(dtype=torch.float16):  # 强制FP16加速
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.2,  # 低随机性，提升格式正确率
                    top_p=0.8,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    num_beams=3,  # 束搜索，优化格式
                    early_stopping=True
                )
        
        # 4. 解码并解析结果
        generated_texts = [
            tokenizer.decode(outputs[i][len(inputs["input_ids"][i]):], skip_special_tokens=True)
            for i in range(len(outputs))
        ]
        
        # 5. 结构化结果
        batch_results = [
            parse_inference_result(generated_texts[i], batch_reviews[i])
            for i in range(len(batch_reviews))
        ]
        results.extend(batch_results)
    
    return results

def single_inference(review):
    """单条评论推理（便于测试）"""
    if not isinstance(review, str) or len(review.strip()) < 5:
        return {
            "review": review,
            "sentiment": None,
            "sentiment_label": -1,
            "key_issue": None,
            "response_to_customer": None,
            "core_coverage": 0.0,
            "status": "failed",
            "error_msg": "评论长度需≥5字"
        }
    # 复用批量推理逻辑
    return batch_inference([review.strip()])[0]

# ====================
# 6. 快速使用示例（可直接调用）
# ====================
if __name__ == "__main__":
    # 示例1：单条评论推理
    print("="*50)
    print("示例1：单条评论推理")
    print("="*50)
    single_review = "菜太咸了，米饭还是凉的，下次不会点了"
    single_result = single_inference(single_review)
    print(f"输入评论：{single_review}")
    print(f"推理结果：")
    if single_result["status"] == "success":
        print(f"  情感倾向：{single_result['sentiment']}（标签：{single_result['sentiment_label']}）")
        print(f"  核心问题：{single_result['key_issue']}")
        print(f"  客服回复：{single_result['response_to_customer']}")
        print(f"  核心词覆盖率：{single_result['core_coverage']:.2%}")
    else:
        print(f"  推理失败：{single_result['error_msg']}")
    
    # 示例2：批量评论推理
    print("\n" + "="*50)
    print("示例2：批量评论推理")
    print("="*50)
    batch_reviews = [
        "菜太咸了，米饭还是凉的，下次不会点了",
        "味道很好，配送也快，会回购",
        "分量很足，但是有点辣，不太能接受",
        "包装很好，保温效果不错，菜也好吃",
        "送得太慢了，菜都凉了，体验很差"
    ]
    batch_results = batch_inference(batch_reviews)
    
    # 打印前3条结果
    print("前3条推理结果：")
    for i, res in enumerate(batch_results[:3]):
        print(f"\n第{i+1}条：")
        print(f"  输入：{res['review']}")
        if res["status"] == "success":
            print(f"  情感：{res['sentiment']} | 回复：{res['response_to_customer']}")
        else:
            print(f"  状态：失败 | 原因：{res['error_msg']}")