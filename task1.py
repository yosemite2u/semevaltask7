# -*- coding: utf-8 -*-
"""
SemEval-2026 Task 7 - Track 2 (MCQ) Inference Script (task1.py)
----------------------------------------------------------
核心机制: Anti-Western Bias Override + Knowledge Generation Prompting
运行环境: Gemini API (gemini-1.5-flash)
工作目录: D:\projecttask1\gemini
核心特性: 支持断点续传 (自动跳过已处理 ID)、指数退避抗并发限流
"""

import os
import pandas as pd
import threading
import time
import shutil
import re
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# ================= 配置参数 =================

# Gemini API 配置
API_KEY = "sk-gG4uXSZUuCk9WCwr0bjLeq6NcfKG5fhsJoTKvXfcGPbwR9AC"
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_NAME = "gemini-1.5-flash"

# Gemini 并发安全甜点值 (太高会触发 429 限流，太低速度慢)
MAX_WORKERS = 15  

# 本地工作目录
BASE_DIR = r"D:\projecttask1\gemini"
INPUT_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "prediction_task1") 
TEMP_TSV_PATH = os.path.join(OUTPUT_DIR, "temp_results.tsv")
FINAL_TSV_PATH = os.path.join(OUTPUT_DIR, "track_2_mcq_prediction.tsv")

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

def extract_answer_from_cot(text):
    if not text: return "A"
    match_tag = re.search(r"\[\[([ABCD])\]\]", text, re.IGNORECASE)
    if match_tag: return match_tag.group(1).upper()
    match_sentence = re.search(r"(?:answer|option) is\s*([ABCD])\b", text, re.IGNORECASE)
    if match_sentence: return match_sentence.group(1).upper()
    matches = re.findall(r"\b([ABCD])\b", text.upper())
    if matches: return matches[-1]
    return "A"

def call_api_with_retry(messages, temp=0.1, retries=6):
    """加入指数退避机制的 API 调用，防止高并发下被封禁"""
    for i in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, stream=False,
                temperature=temp, max_tokens=512, timeout=60
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if i == retries - 1: 
                print(f"[ERROR] API重试耗尽，跳过该题。原因: {e}")
                return None
            # 指数退避: 1s, 2s, 4s, 8s... 极大提升并发情况下的成功率
            time.sleep(2 ** i) 
    return None

def write_result_realtime(result_row):
    with file_lock:
        with open(TEMP_TSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(result_row)

# ================= 核心推理逻辑 =================

def process_mcq_row(row, col_map):
    qid = str(row[col_map['id']])
    question = str(row[col_map['question']])
    
    opt_a = str(row[col_map['A']])
    opt_b = str(row[col_map['B']])
    opt_c = str(row[col_map['C']])
    opt_d = str(row[col_map['D']])
    
    options_text = f"A: {opt_a}\nB: {opt_b}\nC: {opt_c}\nD: {opt_d}"
    region = get_culture_context(qid)
    
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
    
    pred_text = call_api_with_retry([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ], temp=0.1)
    
    choice = extract_answer_from_cot(pred_text)
    mapping = {'A': [1,0,0,0], 'B': [0,1,0,0], 'C': [0,0,1,0], 'D': [0,0,0,1]}
    result_vec = mapping.get(choice, [1,0,0,0])
    
    final_row = [qid] + result_vec
    write_result_realtime(final_row)
    
    return final_row

# ================= 主函数 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    processed_ids = set()
    
    # ---------------- 核心断点续传逻辑 ----------------
    if os.path.exists(TEMP_TSV_PATH):
        try:
            df_temp = pd.read_csv(TEMP_TSV_PATH, sep='\t', usecols=['id'], dtype=str)
            processed_ids = set(df_temp['id'].dropna())
            print(f"[INFO] 扫描到断点记录，已完成 {len(processed_ids)} 题，将自动跳过这些题目。")
        except Exception as e:
            print(f"[WARNING] 临时文件解析失败 ({e})，将清空并重新开始。")
            with open(TEMP_TSV_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(['id', 'A', 'B', 'C', 'D'])
    else:
        with open(TEMP_TSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['id', 'A', 'B', 'C', 'D'])
    # --------------------------------------------------
    
    input_file = os.path.join(INPUT_DIR, "mini_input_5000.tsv")
    
    if not os.path.exists(input_file):
        print(f"[ERROR] 未找到输入文件: {input_file}")
        # 如果是跑另外的文件名，退回尝试读取 mini_input.tsv
        alt_file = os.path.join(INPUT_DIR, "mini_input.tsv")
        if os.path.exists(alt_file):
            print(f"[INFO] 找到替代文件 {alt_file}，使用该文件。")
            input_file = alt_file
        else:
            return

    print(f"[INFO] 启动 SemEval Task 1 (PRO版) 推理测试")
    print(f"[INFO] 模型: {MODEL_NAME} | 设定并发数: {MAX_WORKERS}")

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

    # 过滤出还没跑的题目
    tasks_to_run = [row for _, row in df.iterrows() if str(row[col_map['id']]) not in processed_ids]
    
    if not tasks_to_run:
        print("[INFO] 所有题目均已处理完毕，直接进入文件合并环节。")
    else:
        print(f"[INFO] 本次运行将处理剩余的 {len(tasks_to_run)} 道题...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_mcq_row, row, col_map) for row in tasks_to_run]
            
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Reasoning (Gemini Flash)"):
                pass
            
    print("\n[INFO] 推理阶段完成，正在整合最终预测文件...")

    try:
        df_res = pd.read_csv(TEMP_TSV_PATH, sep='\t', dtype={'id': str})
        
        # 按原题库顺序排序
        try:
            df_res['sort_key'] = df_res['id'].apply(lambda x: (x.split('_')[0], int(x.split('_')[1])))
            df_res = df_res.sort_values('sort_key').drop(columns=['sort_key'])
        except:
            df_res = df_res.sort_values('id')
            
        df_res.to_csv(FINAL_TSV_PATH, sep='\t', index=False)
        print(f"[INFO] 最终预测文件已成功生成: {FINAL_TSV_PATH}")
        
    except Exception as e:
        print(f"[WARNING] 排序与合并步骤出现异常: {e}")
        if os.path.exists(TEMP_TSV_PATH):
            shutil.copy(TEMP_TSV_PATH, FINAL_TSV_PATH)

    print(f"[INFO] 任务运行结束。")

if __name__ == "__main__":
    main()