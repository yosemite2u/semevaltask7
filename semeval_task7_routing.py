# -*- coding: utf-8 -*-
"""
SemEval-2026 Task 7 - Track 2 (MCQ) Inference Script
----------------------------------------------------------
核心机制: 双层动态路由 (Two-Tier Dynamic Routing)
策略说明: 
  - 低资源/易受偏见区域: 采用 Anti-Bias CoT + Self-Consistency (K=3, T=0.5)
  - 高资源主流区域: 采用 Vanilla 直接预测 (T=0.0)，避免过度思考惩罚 (Overthinking Penalty)
"""

import os
import pandas as pd
import threading
import time
import shutil
import re
import zipfile
import csv
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# ================= 配置参数 =================

API_KEY = "sk-142284ab660c44f1980f69db584b56df"
API_BASE = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
MAX_WORKERS = 50  # 异步请求并发数
VOTING_ROUNDS = 3 # Self-Consistency 采样次数

INPUT_DIR = "."
OUTPUT_DIR = "./prediction_task3_routing"
ZIP_NAME = "prediction_task3_routing.zip"
TEMP_TSV_PATH = os.path.join(OUTPUT_DIR, "temp_results.tsv")
FINAL_TSV_PATH = os.path.join(OUTPUT_DIR, "track_2_mcq_prediction.tsv")

# ================= 动态路由规则集合 =================
# 根据验证集消融实验与错误分析划定的低资源/易受偏见文化区域
# 该集合内的区域将被自动路由至 Anti-Bias CoT 复杂推理路径
COMPLEX_REASONING_REGIONS = {
    "am-ET",  # Ethiopia
    "zh-CN",  # China
    "zh-SG",  # Singapore (Chinese Culture)
    "ar-DZ",  # Algeria
    "ar-SA",  # Saudi Arabia
    "as-AS",  # Assam (India)
    "ha-NG"   # Nigeria (Hausa Culture)
}

# ================= 区域文化映射表 =================
CULTURE_MAP = {
    "am-ET": "Ethiopia", "ar-DZ": "Algeria", "ar-EG": "Egypt", 
    "ar-MA": "Morocco", "ar-SA": "Saudi Arabia", "ha-NG": "Nigeria (Hausa Culture)", 
    "fa-IR": "Iran", "az-AZ": "Azerbaijan",
    "as-AS": "Assam (India)", "id-ID": "Indonesia", "ja-JP": "Japan", 
    "ko-KP": "North Korea", "ko-KR": "South Korea", "ms-SG": "Singapore (Malay Culture)", 
    "su-JB": "West Java (Indonesia) - Sundanese Culture", "ta-LK": "Sri Lanka", 
    "ta-SG": "Singapore (Tamil Culture)", "tl-PH": "Philippines", "zh-CN": "China", 
    "zh-SG": "Singapore (Chinese Culture)", "zh-TW": "Taiwan",
    "bg-BG": "Bulgaria", "el-GR": "Greece", "es-ES": "Spain", 
    "eu-PV": "Basque Country (Spain)", "fr-FR": "France", "ga-IE": "Ireland", 
    "sv-SE": "Sweden",
    "es-EC": "Ecuador", "es-MX": "Mexico",
    "en-AU": "Australia", "en-GB": "United Kingdom", "en-US": "United States", 
    "en-CA": "Canada", "en-IE": "Ireland", "en-SG": "Singapore",
    "en-PH": "Philippines", "en-NG": "Nigeria", "en-IN": "India"
}

client = OpenAI(api_key=API_KEY, base_url=API_BASE)
file_lock = threading.Lock() 

# ================= 辅助函数 =================

def get_culture_context(row_id):
    row_id = str(row_id)
    prefix = row_id.split('_')[0]
    if prefix in CULTURE_MAP: 
        return CULTURE_MAP[prefix]
    try:
        if prefix.startswith("en-"):
            parts = prefix.split('-')
            if len(parts) > 1: return f"Region {parts[1]} (English speaking)"
    except: pass
    return "Global Context"

def extract_answer_from_text(text):
    if not text: return "A"
    match_tag = re.search(r"\[\[([ABCD])\]\]", text, re.IGNORECASE)
    if match_tag: return match_tag.group(1).upper()
    match_sentence = re.search(r"(?:answer|option) is\s*([ABCD])\b", text, re.IGNORECASE)
    if match_sentence: return match_sentence.group(1).upper()
    matches = re.findall(r"\b([ABCD])\b", text.upper())
    if matches: return matches[-1]
    return "A"

def call_api_with_retry(messages, temp=0.0, retries=5):
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, stream=False,
                temperature=temp, max_tokens=512, timeout=60
            )
            return response.choices[0].message.content.strip()
        except Exception:
            if i == retries - 1: return None
            time.sleep(1 + i)
    return None

def write_result_realtime(result_row):
    with file_lock:
        with open(TEMP_TSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(result_row)

# ================= 核心推理逻辑 =================

def process_mcq_row(row, col_map):
    qid = str(row[col_map['id']])
    prefix = qid.split('_')[0]
    question = str(row[col_map['question']])
    
    opt_a = str(row[col_map['A']])
    opt_b = str(row[col_map['B']])
    opt_c = str(row[col_map['C']])
    opt_d = str(row[col_map['D']])
    
    options_text = f"A: {opt_a}\nB: {opt_b}\nC: {opt_c}\nD: {opt_d}"
    region = get_culture_context(qid)
    
    # ================= 路由分发机制 =================
    if prefix in COMPLEX_REASONING_REGIONS:
        # 路径 A (Tier 2): 复杂推理策略 (Anti-Bias CoT + Self-Consistency)
        system_prompt = (
            f"You are an elite indigenous cultural anthropologist specializing in {region}. "
            f"Your task is to identify the option that most authentically reflects the historical, traditional, and daily realities of {region}.\n\n"
            f"CRITICAL OVERRIDE (ANTI-WESTERN BIAS): DO NOT project modern Western norms, progressive inclusivity, or globalized hygiene standards onto this context. You must strictly adhere to the authentic local taboos and cultural realities of {region}, even if they contradict modern Euro-American sensibilities.\n\n"
            f"INSTRUCTIONS:\n"
            f"1. KNOWLEDGE ANCHOR: First, explicitly state 2-3 specific traditional facts or taboos in {region} related to the question's topic.\n"
            f"2. EVALUATION: Cross-check each option (A, B, C, D) strictly against the local facts you just stated.\n"
            f"3. CONCLUSION: Select the single best option.\n"
            f"4. FORMATTING: At the very end, output the correct option letter inside double brackets like this: [[A]]."
        )
        user_prompt = f"Question: {question}\nOptions:\n{options_text}\n\nLet's analyze this authentically step by step. Correct Option:"
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        votes = []
        for _ in range(VOTING_ROUNDS):
            pred_text = call_api_with_retry(messages, temp=0.5) 
            votes.append(extract_answer_from_text(pred_text))
            
        final_choice = collections.Counter(votes).most_common(1)[0][0]
        
    else:
        # 路径 B (Tier 1): 基础预测策略 (Vanilla Direct Answer)
        system_prompt = "You are a helpful AI assistant."
        user_prompt = f"Question: {question}\nOptions:\n{options_text}\n\nSelect the correct option. Output ONLY the letter of the correct option inside double brackets, e.g., [[A]]."
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        pred_text = call_api_with_retry(messages, temp=0.0)
        final_choice = extract_answer_from_text(pred_text)

    # 格式化输出
    mapping = {'A': [1,0,0,0], 'B': [0,1,0,0], 'C': [0,0,1,0], 'D': [0,0,0,1]}
    result_vec = mapping.get(final_choice, [1,0,0,0])
    
    final_row = [qid] + result_vec
    write_result_realtime(final_row)
    
    return final_row

# ================= 主函数 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    with open(TEMP_TSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['id', 'A', 'B', 'C', 'D'])
    
    input_file = os.path.join(INPUT_DIR, "mini_input.tsv")
    
    if not os.path.exists(input_file):
        print(f"[ERROR] 未找到输入文件: {input_file}")
        return

    print(f"[INFO] 启动 SemEval Track 2 动态路由推理任务")
    print(f"[INFO] 路由策略: Two-Tier Dynamic Routing | 并发数: {MAX_WORKERS}")

    try:
        df = pd.read_csv(input_file, sep='\t', dtype=str)
        df.columns = df.columns.str.strip()
        
        col_map = {}
        col_map['id'] = next((c for c in df.columns if c.lower() == 'id'), df.columns[0])
        col_map['question'] = next((c for c in df.columns if 'question' in c.lower()), df.columns[1])
        
        def find_opt_col(letter):
            for c in df.columns:
                if re.search(f"option[_ ]?{letter}", c, re.IGNORECASE) or c == letter:
                    return c
            return None

        col_map['A'] = find_opt_col('A') or df.columns[2]
        col_map['B'] = find_opt_col('B') or df.columns[3]
        col_map['C'] = find_opt_col('C') or df.columns[4]
        col_map['D'] = find_opt_col('D') or df.columns[5]
        
    except Exception as e:
        print(f"[ERROR] 解析 TSV 文件失败: {e}")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_mcq_row, row, col_map) for _, row in df.iterrows()]
        
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Reasoning (Dynamic Routing)"):
            pass
            
    print("\n[INFO] 推理阶段完成，正在合并保存最终预测文件...")

    try:
        df_res = pd.read_csv(TEMP_TSV_PATH, sep='\t', dtype={'id': str})
        
        try:
            df_res['sort_key'] = df_res['id'].apply(lambda x: (x.split('_')[0], int(x.split('_')[1])))
            df_res = df_res.sort_values('sort_key').drop(columns=['sort_key'])
        except:
            df_res = df_res.sort_values('id')
            
        df_res.to_csv(FINAL_TSV_PATH, sep='\t', index=False)
        print(f"[INFO] 最终预测文件已生成: {FINAL_TSV_PATH}")
        
        if os.path.exists(TEMP_TSV_PATH):
            os.remove(TEMP_TSV_PATH)
            
    except Exception as e:
        print(f"[WARNING] 排序与合并步骤出现异常: {e}")
        if os.path.exists(TEMP_TSV_PATH):
            shutil.copy(TEMP_TSV_PATH, FINAL_TSV_PATH)

    print(f"[INFO] 任务运行结束。请使用官方 evaluation 脚本计算最终指标。")

if __name__ == "__main__":
    main()
