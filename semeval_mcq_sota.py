# -*- coding: utf-8 -*-
"""
SemEval-2026 Task 7 - Track 2 (MCQ) SOTA Submission Script
----------------------------------------------------------
策略: Zero-Shot Chain-of-Thought (CoT) + 结构化锚点提取
特性: 实时写入硬盘 (防崩溃) + 智能列名识别 + 结果自动排序
合规: 100% 符合比赛规则 (无微调，无 Few-Shot，无外部数据泄露)
"""

import os
import pandas as pd
import threading
import time
import shutil
import re
import zipfile
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# ================= 配置区域 =================

API_KEY = "sk-142284ab660c44f1980f69db584b56df"
API_BASE = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
MAX_WORKERS = 50  # 并发数

# 路径配置
INPUT_DIR = "./input_data"
OUTPUT_DIR = "./prediction"
ZIP_NAME = "prediction_sota.zip"
# 实时保存的临时文件路径
TEMP_TSV_PATH = os.path.join(OUTPUT_DIR, "temp_results.tsv")
# 最终提交的标准文件路径
FINAL_TSV_PATH = os.path.join(OUTPUT_DIR, "track_2_mcq_prediction.tsv")

# ================= 全量文化映射表 =================
# 覆盖 Track 2 所有可能的 ID 前缀
CULTURE_MAP = {
    # 非洲与中东
    "am-ET": "Ethiopia", "ar-DZ": "Algeria", "ar-EG": "Egypt", 
    "ar-MA": "Morocco", "ar-SA": "Saudi Arabia", "ha-NG": "Nigeria (Hausa Culture)", 
    "fa-IR": "Iran", "az-AZ": "Azerbaijan",
    # 亚洲
    "as-AS": "Assam (India)", "id-ID": "Indonesia", "ja-JP": "Japan", 
    "ko-KP": "North Korea", "ko-KR": "South Korea", "ms-SG": "Singapore (Malay Culture)", 
    "su-JB": "West Java (Indonesia) - Sundanese Culture", "ta-LK": "Sri Lanka", 
    "ta-SG": "Singapore (Tamil Culture)", "tl-PH": "Philippines", "zh-CN": "China", 
    "zh-SG": "Singapore (Chinese Culture)", "zh-TW": "Taiwan",
    # 欧洲
    "bg-BG": "Bulgaria", "el-GR": "Greece", "es-ES": "Spain", 
    "eu-PV": "Basque Country (Spain)", "fr-FR": "France", "ga-IE": "Ireland", 
    "sv-SE": "Sweden",
    # 美洲
    "es-EC": "Ecuador", "es-MX": "Mexico",
    # 英语区 (en-XX)
    "en-AU": "Australia", "en-GB": "United Kingdom", "en-US": "United States", 
    "en-CA": "Canada", "en-IE": "Ireland", "en-SG": "Singapore",
    "en-PH": "Philippines", "en-NG": "Nigeria", "en-IN": "India"
}

client = OpenAI(api_key=API_KEY, base_url=API_BASE)
file_lock = threading.Lock() # 用于多线程写入文件锁

# ================= 核心工具函数 =================

def get_culture_context(row_id):
    """根据 ID 解析具体的文化背景"""
    row_id = str(row_id)
    prefix = row_id.split('_')[0]
    
    # 精确匹配
    if prefix in CULTURE_MAP: 
        return CULTURE_MAP[prefix]
    
    # 模糊匹配 (处理 en-XX)
    try:
        if prefix.startswith("en-"):
            parts = prefix.split('-')
            if len(parts) > 1: return f"Region {parts[1]} (English speaking)"
    except: pass
    return "Global Context"

def extract_answer_from_cot(text):
    """
    鲁棒性答案提取器
    优先级: [[A]] > "Answer is A" > 最后一个出现的选项
    """
    if not text: return "A" # 兜底
    
    # 1. 锚点提取 (Prompt 强制要求的格式)
    match_tag = re.search(r"\[\[([ABCD])\]\]", text, re.IGNORECASE)
    if match_tag: return match_tag.group(1).upper()
    
    # 2. 句式提取
    match_sentence = re.search(r"(?:answer|option) is\s*([ABCD])\b", text, re.IGNORECASE)
    if match_sentence: return match_sentence.group(1).upper()
    
    # 3. 单词提取 (取最后出现的字母)
    matches = re.findall(r"\b([ABCD])\b", text.upper())
    if matches: return matches[-1]
    
    return "A"

def call_api_with_retry(messages, temp=0.1, retries=5):
    """API 调用封装，增加重试次数以应对网络波动"""
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, stream=False,
                temperature=temp, max_tokens=512, timeout=60
            )
            return response.choices[0].message.content.strip()
        except Exception:
            if i == retries - 1: return None
            time.sleep(1 + i) # 递增等待
    return None

def write_result_realtime(result_row):
    """线程安全地写入一行结果"""
    with file_lock:
        with open(TEMP_TSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(result_row)

# ================= Track 2: MCQ 逻辑 =================

def process_mcq_row(row, col_map):
    """单行数据处理逻辑"""
    qid = str(row[col_map['id']])
    question = str(row[col_map['question']])
    
    # 获取选项内容
    opt_a = str(row[col_map['A']])
    opt_b = str(row[col_map['B']])
    opt_c = str(row[col_map['C']])
    opt_d = str(row[col_map['D']])
    
    options_text = f"A: {opt_a}\nB: {opt_b}\nC: {opt_c}\nD: {opt_d}"
    region = get_culture_context(qid)
    
    # SOTA Prompt: 强制 CoT 思考 + 结构化输出
    system_prompt = (
        f"You are a cultural expert specializing in {region}. "
        f"Your task is to identify the option that best reflects the common cultural practices, values, or daily life in {region}.\n\n"
        f"INSTRUCTIONS:\n"
        f"1. Analyze the question in the context of {region}.\n"
        f"2. Evaluate each option (A, B, C, D) step-by-step.\n"
        f"3. Conclude with the single best option.\n"
        f"4. CRITICAL: At the very end of your response, output the correct option letter inside double brackets like this: [[A]]."
    )
    
    user_prompt = f"Question: {question}\nOptions:\n{options_text}\n\nLet's think step by step. Correct Option:"
    
    # 执行推理
    pred_text = call_api_with_retry([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ], temp=0.1)
    
    # 提取答案
    choice = extract_answer_from_cot(pred_text)
    
    # 转 One-Hot
    mapping = {'A': [1,0,0,0], 'B': [0,1,0,0], 'C': [0,0,1,0], 'D': [0,0,0,1]}
    result_vec = mapping.get(choice, [1,0,0,0])
    
    # 组合结果并实时写入
    final_row = [qid] + result_vec
    write_result_realtime(final_row)
    
    return final_row

# ================= 主流程 =================

def main():
    # 1. 初始化目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 2. 准备实时保存文件 (写入表头)
    # 如果文件已存在，则覆盖（如果想断点续传，需修改此处逻辑，这里假设每次重跑）
    with open(TEMP_TSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['id', 'A', 'B', 'C', 'D'])
    
    input_file = os.path.join(INPUT_DIR, "track_2_mcq_input.tsv")
    
    if not os.path.exists(input_file):
        print(f"❌ 错误：未找到 {input_file}")
        return

    print(f"🚀 启动 SemEval Track 2 SOTA 推理 (实时存档版)")
    print(f"模式: Zero-Shot CoT | 并发: {MAX_WORKERS}")

    # 3. 智能读取数据 (解决列名空格问题)
    try:
        # 读取全部列为字符串
        df = pd.read_csv(input_file, sep='\t', dtype=str)
        
        # 🔥 核心修复：去除列名首尾空格
        df.columns = df.columns.str.strip()
        
        # 动态列名映射
        col_map = {}
        # 查找 ID
        col_map['id'] = next((c for c in df.columns if c.lower() == 'id'), df.columns[0])
        # 查找 Question
        col_map['question'] = next((c for c in df.columns if 'question' in c.lower()), df.columns[1])
        
        # 查找选项列 (兼容 'option A' 和 'option_A' 等)
        def find_opt_col(letter):
            for c in df.columns:
                # 匹配 "option A", "option_A", "A", "Option A" 等
                if re.search(f"option[_ ]?{letter}", c, re.IGNORECASE) or c == letter:
                    return c
            return None # 此时会在下面的逻辑中报错或兜底

        col_map['A'] = find_opt_col('A') or df.columns[2]
        col_map['B'] = find_opt_col('B') or df.columns[3]
        col_map['C'] = find_opt_col('C') or df.columns[4]
        col_map['D'] = find_opt_col('D') or df.columns[5]
        
    except Exception as e:
        print(f"❌ 读取 TSV 失败: {e}")
        return

    # 4. 并发执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_mcq_row, row, col_map) for _, row in df.iterrows()]
        
        # 使用 Tqdm 显示进度
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Reasoning"):
            pass
            
    print("\n✅ 所有任务已完成，正在整理最终文件...")

    # 5. 后处理：排序与重写
    # 因为多线程写入导致 TEMP 文件中的行是乱序的，这里重新读取并按 ID 排序
    try:
        df_res = pd.read_csv(TEMP_TSV_PATH, sep='\t', dtype={'id': str})
        
        # 尝试智能排序 (ID前缀_数字)
        try:
            df_res['sort_key'] = df_res['id'].apply(lambda x: (x.split('_')[0], int(x.split('_')[1])))
            df_res = df_res.sort_values('sort_key').drop(columns=['sort_key'])
        except:
            df_res = df_res.sort_values('id')
            
        # 保存为最终符合规范的文件
        df_res.to_csv(FINAL_TSV_PATH, sep='\t', index=False)
        print(f"📄 最终文件已生成: {FINAL_TSV_PATH}")
        
        # 删除临时文件 (可选)
        if os.path.exists(TEMP_TSV_PATH):
            os.remove(TEMP_TSV_PATH)
            
    except Exception as e:
        print(f"⚠️ 排序步骤出错 (不影响提交): {e}")
        # 如果排序失败，TEMP 文件依然可用，只需改名
        if os.path.exists(TEMP_TSV_PATH):
            shutil.copy(TEMP_TSV_PATH, FINAL_TSV_PATH)

    # 6. 打包 ZIP
    print("📦 正在打包提交文件...")
    with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 严格遵守官方目录结构: prediction/track_2_mcq_prediction.tsv
        zf.write(FINAL_TSV_PATH, arcname=os.path.join("prediction", "track_2_mcq_prediction.tsv"))
                
    print(f"🎉 提交包已就绪！")
    print(f"👉 文件位置: {os.path.abspath(ZIP_NAME)}")
    print("👉 请上传至 Codabench -> Evaluation Phase -> Track 2")

if __name__ == "__main__":
    main()