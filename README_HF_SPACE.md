---
title: SmolLM2-135M
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: apache-2.0
---

# SmolLM2-135M Text Generation Demo

A lightweight language model (135M parameters) fine-tuned exclusively on Shakespeare's **Coriolanus**. The model writes in the style of a dramatic play, complete with character names, stage directions, and Shakespearean dialogue.

## Features

- **135M Parameters** - Efficient and fast inference
- **Grouped Query Attention (GQA)** - Optimized attention mechanism
- **Flash Attention** - Fast attention computation
- **Interactive UI** - Easy-to-use Gradio interface

## Model Architecture

- **Hidden Size:** 576
- **Layers:** 30
- **Attention Heads:** 9 (3 KV heads)
- **Vocabulary:** 49,152 tokens
- **Max Context:** 8,192 tokens

## Usage

1. Enter your prompt in the text box (try prompts like "CORIOLANUS:" or "Enter CORIOLANUS and MENENIUS")
2. Adjust generation parameters (temperature, top-p, etc.)
3. Click "Generate" to create text in the style of a dramatic play

## Parameters

- **Temperature:** Controls randomness (lower = more focused)
- **Top-p:** Nucleus sampling threshold
- **Top-k:** Limits to top k tokens
- **Repetition Penalty:** Reduces repetition (higher = less repetition)

