# -*- coding: utf-8 -*-
"""
SemEval-2026 Task 7 - Track 2 (MCQ) Voting Script (task2_voting.py)
----------------------------------------------------------
提分核心: Self-Consistency (多数表决) + Anti-Western Bias Prompt
特性: 每道题在 temp=0.5 下独立思考 3 次，取众数作为最终答案
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

# ================= 配置区域 =================

API_KEY = "sk-142284ab660c44f1980f69db584b56df"
API_BASE = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
MAX_WORKERS = 50  # 并发数
VOTING_ROUNDS = 3 # 每道题思考/投票的次数 (经典值: 3 或 5)

INPUT_DIR = "."
OUTPUT_DIR = "./prediction_task2_voting"  # 独立输出目录
ZIP_NAME = "prediction_task2_voting.zip"
TEMP_TSV_PATH = os.path.join(OUTPUT_DIR, "temp_results.tsv")
FINAL_TSV_PATH = os.path.join(OUTPUT_DIR, "track_2_mcq_prediction.tsv")

# ================= 全量文化映射表 =================
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

# ================= 核心工具函数 =================

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

def extract_answer_from_cot(text):
    if not text: return "A"
    match_tag = re.search(r"\[\[([ABCD])\]\]", text, re.IGNORECASE)
    if match_tag: return match_tag.group(1).upper()
    match_sentence = re.search(r"(?:answer|option) is\s*([ABCD])\b", text, re.IGNORECASE)
    if match_sentence: return match_sentence.group(1).upper()
    matches = re.findall(r"\b([ABCD])\b", text.upper())
    if matches: return matches[-1]
    return "A"

def call_api_with_retry(messages, temp=0.5, retries=5):
    """注意：此处的默认 temperature 已改为 0.5 以增加思维多样性"""
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

# ================= Track 2: MCQ 多数表决逻辑 =================

def process_mcq_row(row, col_map):
    qid = str(row[col_map['id']])
    question = str(row[col_map['question']])
    
    opt_a = str(row[col_map['A']])
    opt_b = str(row[col_map['B']])
    opt_c = str(row[col_map['C']])
    opt_d = str(row[col_map['D']])
    
    options_text = f"A: {opt_a}\nB: {opt_b}\nC: {opt_c}\nD: {opt_d}"
    region = get_culture_context(qid)
    
    # 保留 PRO 版的极品 Prompt
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
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 🔥 核心提分点: 多次推理与多数表决
    votes = []
    for _ in range(VOTING_ROUNDS):
        pred_text = call_api_with_retry(messages, temp=0.5) # 提高 temp，增加探索域
        choice = extract_answer_from_cot(pred_text)
        votes.append(choice)
        
    # 统计得票数最高的结果
    # most_common(1) 返回类似 [('A', 2)]，[0][0] 获取第一名也就是 'A'
    final_choice = collections.Counter(votes).most_common(1)[0][0]
    
    mapping = {'A': [1,0,0,0], 'B': [0,1,0,0], 'C': [0,0,1,0], 'D': [0,0,0,1]}
    result_vec = mapping.get(final_choice, [1,0,0,0])
    
    final_row = [qid] + result_vec
    write_result_realtime(final_row)
    
    return final_row

# ================= 主流程 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    with open(TEMP_TSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['id', 'A', 'B', 'C', 'D'])
    
    input_file = os.path.join(INPUT_DIR, "mini_input.tsv")
    
    if not os.path.exists(input_file):
        print(f"❌ 错误：未找到 {input_file}")
        return

    print(f"🚀 启动 SemEval Track 2 Voting 级推理 (task2_voting.py)")
    print(f"模式: Self-Consistency ({VOTING_ROUNDS} votes) | 并发: {MAX_WORKERS}")

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
        print(f"❌ 读取 TSV 失败: {e}")
        return

    # 因为每道题要调 3 次 API，整体进度条会变慢，属于正常现象
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_mcq_row, row, col_map) for _, row in df.iterrows()]
        
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Reasoning (Voting)"):
            pass
            
    print("\n✅ 所有投票计算已完成，正在整理最终文件...")

    try:
        df_res = pd.read_csv(TEMP_TSV_PATH, sep='\t', dtype={'id': str})
        
        try:
            df_res['sort_key'] = df_res['id'].apply(lambda x: (x.split('_')[0], int(x.split('_')[1])))
            df_res = df_res.sort_values('sort_key').drop(columns=['sort_key'])
        except:
            df_res = df_res.sort_values('id')
            
        df_res.to_csv(FINAL_TSV_PATH, sep='\t', index=False)
        print(f"📄 最终文件已生成: {FINAL_TSV_PATH}")
        
        if os.path.exists(TEMP_TSV_PATH):
            os.remove(TEMP_TSV_PATH)
            
    except Exception as e:
        print(f"⚠️ 排序步骤出错: {e}")
        if os.path.exists(TEMP_TSV_PATH):
            shutil.copy(TEMP_TSV_PATH, FINAL_TSV_PATH)

    print(f"🎉 task2 (Voting版) 跑分已完成，请使用 eval 脚本验证分数。")

if __name__ == "__main__":
    main()