# SemEval-2026 Task 7: Two-Tier Dynamic Routing Framework

This repository contains the official code for the paper: 
**"uir-cis-7 at SemEval-2026 Task 7: Zero-Shot Chain-of-Thought Reasoning for Cross-Cultural Daily Knowledge"**.

## 📌 Overview
To address Western-centric bias and the "overthinking penalty" in Large Language Models (LLMs), we propose a **Two-Tier Dynamic Routing Framework**. Based on the cultural resource density of the query's geographic region, our system intelligently dispatches queries to either a direct-answer pathway (Vanilla) or an Anti-Bias Persona-Conditioned Chain-of-Thought pathway.

Our system achieved **89.02%** overall macro-average accuracy on the official SemEval leaderboard.

## 🚀 Repository Structure
- `task1_clean.py`: Baseline script using the direct prompt (Tier 1 globally).
- `task2_voting.py`: Baseline script using Anti-Bias Persona CoT and Self-Consistency voting (Tier 2 globally).
- `task3_routing.py`: **Main Contribution.** The dynamic routing framework integrating both pathways.
- `qwentask3.py`: Cross-model validation script utilizing Qwen2.5-72B to prove our routing strategy is model-agnostic.

## 📝 Citation
If you find our code or methodology useful, please cite our paper.
