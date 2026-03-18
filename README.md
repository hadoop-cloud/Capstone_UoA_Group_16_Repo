# RAG-Integrated Self-Play Conversational Fine-Tuning Extension for LlamaFactory

## Overview
This project extends the **LlamaFactory** framework with a **retrieval-augmented self-play fine-tuning pipeline** for **medical dialogue generation**. The goal is to improve factual grounding and reduce hallucinations in medical responses by combining:

- **LoRA-based supervised fine-tuning**
- **Retrieval-Augmented Generation (RAG)**
- **Role-based self-play data generation**
- **Guardrails and critic-based filtering**

The system is implemented in two stages:

1. **Baseline model (M1):** Fine-tuned on curated medical datasets  
2. **Enhanced model (M2):** Continued fine-tuning of M1 using filtered synthetic, evidence-grounded self-play data

---

## Motivation
Large language models are powerful, but in healthcare they may generate **hallucinated or unsupported responses**. This project addresses that challenge by grounding generated answers in retrieved biomedical evidence and using self-play to create additional high-quality training data.

This work directly aligns with the future direction suggested in the LlamaFactory paper:
> exploring stronger fine-tuning methods for conversational models, e.g., self-play

---

## Project Objectives
- Extend LlamaFactory with a **retrieval-grounded self-play pipeline**
- Improve **factual reliability** in medical dialogue generation
- Reduce unsupported claims using **evidence-based response generation**
- Evaluate improvements using **CCR** and **UCR**

---

## Architecture

### Stage 1: Baseline Fine-Tuning (M1)
- Base model: `Llama-3.2-3B-Instruct`
- Framework: `LlamaFactory`
- Fine-tuning method: `LoRA`
- Training data: curated medical dataset

### Stage 2: RAG + Self-Play Augmentation (M2)
- Build medical evidence corpus
- Chunk and embed text
- Store embeddings in `Chroma`
- Retrieve top-k evidence chunks
- Generate synthetic data using:
  - **Questioner**
  - **Assistant**
  - **Critic**
- Apply guardrails and filtering
- Continue fine-tuning from M1 to produce M2

---

## Project Workflow

```text
Medical Dataset
   ↓
LlamaFactory SFT with LoRA
   ↓
Baseline Model M1
   ↓
Medical Evidence Corpus
   ↓
Chunking + Embeddings + Chroma DB
   ↓
RAG Retrieval
   ↓
Self-Play Generation
   ↓
Guardrails + Critic Filtering
   ↓
Synthetic Dataset
   ↓
Second-Stage Fine-Tuning
   ↓
Enhanced Model M2
   ↓
Evaluation (CCR / UCR)
