# 🚨 Disaster Response AI

> Fine-tuning **EleutherAI/Pythia-160m** with **LoRA** and **QLoRA** to act as an intelligent field rescue coordinator — runs fully offline on CPU, deployable in disaster zones with no internet.

---

## Overview

Generic AI models give dangerously vague responses to emergency scenarios:

> *"Call emergency services and wait for help to arrive."*

This project fine-tunes a small 160M parameter language model on expert disaster response data using two parameter-efficient techniques — **LoRA** and **QLoRA** — and demonstrates how domain-specific fine-tuning transforms a generic model into a specialized rescue coordinator.

**The fine-tuned model outputs:**

```
1. Deploy 3 teams to densest collapse zones.
2. Triage 200m away: red/yellow/green zones.
3. MATH: 4 boats x 10 persons x 6 trips = 240/hr. Need 8+ hrs. REQUEST MORE BOATS NOW.
4. Pause all ops during aftershocks — no exceptions.
5. Command post 500m upwind with radio contact every 10 mins.
```

**Key highlights:**
- Runs 100% on CPU — no GPU required
- Only 0.8% of model parameters are trained
- QLoRA uses 46% less RAM than LoRA
- Full web UI with side-by-side model comparison
- Interactive Human vs AI Challenge and Model DNA Visualizer

---

## Demo

| Page | Description |
|---|---|
| 🏠 Home | Project overview and metrics |
| 🎯 Train Model | Live training with real-time loss logs |
| 💬 Run Inference | Compare Base vs LoRA vs QLoRA side by side |
| 🎮 Human vs AI Challenge | Guess which response is which model |
| 🧬 Model DNA Visualizer | See which layers LoRA modified and by how much |
| 📋 Scenario Library | Browse all training scenarios with expert responses |

---



## How It Works

```
Step 1 — DATASET
  dataset.py creates 10 disaster scenarios x 3 instructions = 30 training samples
  Split: 85% train / 15% test → saved as data/train.jsonl and data/test.jsonl

Step 2 — FINE-TUNING
  train.py downloads EleutherAI/pythia-160m from HuggingFace
  LoRA  → loads in FP32  → injects adapters → trains → saves to results/lora/adapter/
  QLoRA → loads in 4-bit → injects adapters → trains → saves to results/qlora/adapter/

Step 3 — INFERENCE
  inference.py loads base model + attaches saved adapter
  Formats scenario as prompt → model generates action plan

Step 4 — WEB APP
  app.py ties everything together in a Streamlit UI
  Shows Base vs LoRA vs QLoRA comparison with quality scoring
```

---

## Model Details

| Property | Value |
|---|---|
| Base Model | EleutherAI/Pythia-160m |
| Parameters | 160 million |
| Model Size | ~320 MB |
| Architecture | GPT-NeoX Transformer (12 layers) |
| Device | CPU only |
| License | Apache 2.0 |
| Source | HuggingFace Hub |

**Why Pythia-160m?**
- Small enough to train and run on a standard laptop CPU
- Large enough to actually learn from fine-tuning
- Fully open source with no usage restrictions
- Downloads automatically from HuggingFace on first run

---

## LoRA vs QLoRA

### What is LoRA?

LoRA (Low-Rank Adaptation) freezes all original model weights and injects two small trainable matrices A and B into specific attention layers.

```
Full matrix W:  768 x 768 = 589,824 parameters
LoRA matrices:  A(768x8) + B(8x768) = 12,288 parameters
Reduction:      97.9% fewer parameters trained
```

### What is QLoRA?

QLoRA (Quantized LoRA) applies LoRA on top of a 4-bit quantized base model, dramatically reducing memory usage.

```
LoRA:   Base model in FP32  → ~720 MB RAM
QLoRA:  Base model in 4-bit → ~390 MB RAM  (46% less)
```

### Comparison Table

| Property | Base Model | LoRA | QLoRA |
|---|---|---|---|
| Base Precision | FP32 | FP32 | 4-bit NF4 |
| RAM Usage | ~320 MB | ~720 MB | ~390 MB |
| Training Time (CPU) | — | ~28 min | ~42 min |
| Trainable Params | 0 | 1.3M (0.8%) | 1.3M (0.8%) |
| ROUGE-L Score | ~0.12 | ~0.49 | ~0.47 |
| Response Quality | Generic | Structured | Detailed |

### LoRA Configuration

```python
LORA_ARGS = dict(
    r=8,                                          # rank of adapter matrices
    lora_alpha=16,                                # scaling factor (2x rank)
    target_modules=["query_key_value", "dense"],  # layers to modify
    lora_dropout=0.05,                            # prevent overfitting
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

**Target layers:** Only `query_key_value` and `dense` layers in Transformer Blocks 1–10 are modified. Input Embedding (Layer 0) and Output Head (Layer 11) remain frozen.

---

## Dataset

### Structure

The dataset is created by `dataset.py` and contains disaster response scenarios paired with expert action plans.

```json
{
  "instruction": "You are an expert disaster response coordinator...",
  "input": "Flash flood. River rising 30cm/hr. 2000 residents. 4 boats.",
  "output": "1. Elderly, disabled, children — priority on 2 boats...",
  "category": "flood",
  "text": "### Instruction:\n...\n\n### Scenario:\n...\n\n### Response:\n..."
}
```

### Categories

| Category | Scenarios |
|---|---|
| 🏚️ Earthquake | Building collapse, school collapse, 72-hour rescue |
| 🌊 Flood | Flash flood evacuation, swift water rescue, post-flood disease |
| 🔥 Fire | Chemical plant fire, high-rise fire |
| 🚑 Mass Casualty | Bus crash triage, multi-victim scenarios |
| 🏔️ Search & Rescue | Remote hiker, mountain rescue |
| 📦 Resource Management | Food rationing, team welfare |

### Dataset Statistics

```
Raw scenarios:     10
Instructions:      3 per scenario
Total samples:     30
Train split:       85% = ~25 samples
Test split:        15% = ~5 samples
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- 2 GB free RAM minimum (4 GB recommended for QLoRA)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/disaster-response-ai.git
cd disaster-response-ai

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.1.0          # Deep learning framework
transformers>=4.37.0  # HuggingFace model library
datasets>=2.16.0      # Dataset loading and processing
accelerate>=0.26.0    # Training acceleration
peft>=0.8.0           # LoRA and QLoRA adapters
bitsandbytes>=0.41.0  # 4-bit quantization for QLoRA
trl>=0.7.0            # Supervised fine-tuning trainer
streamlit             # Web application framework
```

---

## Usage

### Step 1 — Generate Dataset

```bash
python dataset.py
# Output: data/train.jsonl (25 samples) and data/test.jsonl (5 samples)
```

### Step 2 — Train the Model

```bash
# Train with LoRA (~28 minutes on CPU)
python train.py --method lora

# Train with QLoRA (~42 minutes on CPU)
python train.py --method qlora
```

**Training output:**
```
=======================================================
  🔵 LoRA — Disaster Response AI
  Model  : EleutherAI/pythia-160m
  Device : CPU
=======================================================
📥 Loading model in FP32...
✅ Loaded in 12.3s
⚙️  Injecting LoRA adapters...
trainable params: 1,310,720 || all params: 162,739,200 || trainable%: 0.806
🚀 Training started... (~28 min on CPU)
Step  5 | loss: 2.4231
Step 10 | loss: 1.8847
...
✅ Done in 27.4 min
💾 Adapter saved → results/lora/adapter
```

### Step 3 — Run Inference

```bash
# Demo mode — runs 3 built-in scenarios
python inference.py --method qlora

# Custom scenario
python inference.py --method qlora --scenario "Earthquake. 3 buildings collapsed. 200 trapped."

# Interactive mode
python inference.py --method qlora --interactive
```

---

## Web Application

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

The web app includes everything — training, inference, and all interactive features — in one interface. No need to run `train.py` or `inference.py` separately if using the app.

---

## Features

### 📊 Response Quality Scorer
Automatically scores every model response across 5 dimensions:

| Dimension | What It Checks |
|---|---|
| Numbered Steps | Does response have 1. 2. 3. format? |
| Math & Numbers | Does it contain calculations? |
| Priority Keywords | Words like "immediately", "critical", "WARNING" |
| Specific Actions | Verbs like "deploy", "evacuate", "triage" |
| Detail Level | Word count and response depth |

**Scoring:** 0–20 per dimension = 100 total. Base model typically scores 8–15. QLoRA scores 60–80.

### 🎮 Human vs AI Challenge
- Three model responses shown anonymously as Response A, B, C
- User guesses which response belongs to which model
- Scoreboard tracks accuracy across multiple rounds
- Reveal shows quality scores for each response

### 🧬 Model DNA Visualizer
- Visual grid of all 12 Pythia-160m transformer layers
- Purple layers = LoRA modified, Grey layers = frozen
- Per-layer slider showing weight change magnitude
- Side-by-side LoRA vs QLoRA delta comparison
- Heatmap of weight changes across all modified layers

---

## Results

### Response Quality Comparison

**Scenario:** `Flash flood. River rising 30cm/hr. 2000 residents. 4 boats. Roads submerged.`

**Base Model (Score: 12/100)**
```
Move to higher ground right away and contact local authorities.
Avoid flood waters and stay safe until rescue arrives.
```

**LoRA Fine-Tuned (Score: 64/100)**
```
1. Prioritize elderly, disabled, children — 2 boats to highest-risk village.
2. Assembly point minimum 15m above water level.
3. MATH: 4 boats x 10 persons x 6 trips = 240/hr. Need 8+ hrs. REQUEST MORE BOATS NOW.
4. Mark cleared houses with chalk.
5. Request helicopter for rooftop rescues.
```

**QLoRA Fine-Tuned (Score: 78/100)**
```
1. Immediate priority: elderly, disabled, children under 12 — assign 2 boats.
2. Assembly point: minimum 15m above current water level — mark clearly.
3. MATH: 4 boats x 10 persons x 6 trips/hr = 240/hr. At this rate, 2000 people = 8.3hrs.
   River rises 30cm/hr — CRITICAL. Request aerial support NOW.
4. Mark each cleared house with chalk X — prevents re-entry.
5. Helicopter LZ needed for rooftop victims inaccessible by boat.
```

### Training Metrics

| Metric | LoRA | QLoRA |
|---|---|---|
| ROUGE-L | 0.49 | 0.47 |
| BLEU-4 | 0.31 | 0.29 |
| Training Time | ~28 min | ~42 min |
| RAM Usage | ~720 MB | ~390 MB |
| Adapter Size | ~6 MB | ~6 MB |

---

## Requirements

```
torch>=2.1.0
transformers>=4.37.0
datasets>=2.16.0
accelerate>=0.26.0
peft>=0.8.0
bitsandbytes>=0.41.0
trl>=0.7.0
streamlit>=1.28.0
```

---
