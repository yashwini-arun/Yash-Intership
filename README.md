# 🛡️ ToxicShield — LoRA vs QLoRA Social Media Comment Detoxifier


> A machine learning web application that compares **LoRA** and **QLoRA** fine-tuning techniques by transforming toxic social media comments into positive, constructive messages — complete with personality detection and fake social media post generation.

---

## 🧠 What is ToxicShield?

ToxicShield is a Python-based web application built with **Streamlit** that demonstrates the practical difference between two popular fine-tuning techniques in modern NLP:

| Technique | Description |
|-----------|-------------|
| **LoRA** | Low-Rank Adaptation — adds small trainable adapter matrices (~6MB) on top of a frozen large model |
| **QLoRA** | Quantized LoRA — same as LoRA but the base model is compressed to use less memory |

The app takes any **toxic or negative comment** as input and produces two different positive rewrites — one from each model — displayed side by side. It also detects the likely **personality behind the toxic comment** and renders the results as realistic **Instagram and Twitter post cards**.

---

## ✨ Live Demo Features

### Tab 1 — Try It Live
- Paste any toxic comment into the text box
- Click **Generate Positive Rewrites**
- See LoRA (empathetic style) and QLoRA (motivational style) outputs side by side
- 3 preset example buttons for quick testing
- Inference time shown for each model

### Tab 2 — Personality + Social Card
- **Personality Detection** — identifies who likely wrote the toxic comment
  - 😤 Frustrated Overachiever
  - 🎮 Competitive Gamer
  - 😔 Burnt-Out Professional
  - 😠 Angry Parent / Authority Figure
  - 💔 Secretly Insecure Person
  - 📰 Passionate Debater
  - 🌋 Stressed & Venting
- **Instagram Card** — LoRA rewrite shown as a real-looking Instagram post
- **Twitter/X Card** — QLoRA rewrite shown as a real-looking tweet

### Tab 3 — Batch Compare
- Runs 10 preset toxic samples through both models automatically
- Shows personality + both rewrites for each sample
- Download all results as a **CSV file**

### Tab 4 — Metrics
- Model size comparison (BART 550MB vs T5 240MB)
- Parameter count comparison (140M vs 60M)
- Architecture comparison table
- Bar charts for visual comparison

---

## ⚙️ LoRA vs QLoRA — Key Concepts

```
PRE-TRAINING  (done by Meta / Google — takes months on thousands of GPUs)
══════════════════════════════════════════════════════════════════════════
Billions of internet sentences
        ↓  train
facebook/opt-350m  →  Knows English, can write text
                       BUT doesn't know detoxification

FINE-TUNING  (done by researchers — takes hours on 1 GPU)
══════════════════════════════════════════════════════════
ParaDetox dataset (19,000 toxic → clean pairs)
        ↓  fine-tune with LoRA or QLoRA
Detoxification model  →  Knows how to rewrite toxic text
```

### LoRA
```
Base Model (float32, full size ~700MB)
        +
Small Adapter (~6MB trainable weights)
        =
LoRA Model  ✅
```

### QLoRA
```
Base Model (compressed, layer-by-layer loading)
        +
Same Small Adapter (~6MB trainable weights)
        =
QLoRA Model  ✅  (uses less RAM)
```

> **Key Insight:** The adapter is identical in both. The only difference is HOW the base model is loaded into memory.

---

## 🤖 Models Used

### In app.py (Main Web Application)

| Role | Model ID | Architecture | Parameters | Size | Trained On |
|------|----------|-------------|-----------|------|-----------|
| **LoRA Model** | `s-nlp/bart-base-detox` | BART-base | 140M | ~550MB | ParaDetox |
| **QLoRA Model** | `erfansadraiye/detoxify` | T5-Small | 60M | ~240MB | ParaDetox |

### In step1 & step2 (Learning / Demo Files)

| Role | Model ID | Size | Purpose |
|------|----------|------|---------|
| **Base Model** | `facebook/opt-350m` | ~700MB | Meta's OPT language model — general text generation |
| **LoRA Adapter** | `ybelkada/opt-350m-lora` | ~6MB | Official PEFT example adapter from HuggingFace docs |

> ⚠️ **Important:** Both LoRA and QLoRA in step1/step2 use the **same adapter** (`ybelkada/opt-350m-lora`). This is intentional — LoRA vs QLoRA is about how the base model is loaded, NOT about the adapter.

### Why Two Different Sets of Models?

| Files | Models | Reason |
|-------|--------|--------|
| step1, step2, step3 | OPT-350m + ybelkada adapter | Learning exercise — demonstrates LoRA/QLoRA concept simply |
| app.py | BART-detox + T5-detox | Actual detoxification — these models are properly trained for the task |

---

## 🛠️ Installation & Setup

### Prerequisites
- Windows 10 or 11
- Python 3.11 installed and added to PATH
- VS Code (recommended)
- ~3GB free disk space (for model downloads)
- Internet connection (for first run)

### Step 1 — Clone or Download the Project
```
Download all project files to:
C:\Users\YourName\Desktop\toxicshield\
```

### Step 2 — Run Setup (ONE TIME ONLY)
Open PowerShell or VS Code terminal inside the `toxicshield` folder:
```powershell
.\setup.bat
```
This will:
- Create `venv/` folder
- Install PyTorch 2.2.0 (CPU)
- Install all 11 libraries
- Takes approximately 3-5 minutes

---

## ▶️ How to Run

Every time you open a new terminal, follow these steps:

### Step 1 — Activate Virtual Environment
```powershell
.\venv\Scripts\activate
```
You should see `(venv)` appear at the start of the terminal line.

### Step 2 — Download LoRA Model (First time only, ~2 min)
```powershell
python src/step1_download_lora.py
```

### Step 3 — Download QLoRA Model (First time only, ~1 min)
```powershell
python src/step2_download_qlora.py
```

### Step 4 — Run Comparison (Optional)
```powershell
python src/step3_compare.py
```
Saves CSV and chart to `outputs/` folder.

### Step 5 — Launch Web App
```powershell
streamlit run app.py
```
Opens automatically at **http://localhost:8501**

> 💡 On first launch, app.py will download BART (~550MB) and T5 (~240MB). This takes 3-5 minutes. After that, models are cached and load in seconds.

---

## 🔄 How It Works — End to End

```
User types toxic comment
        ↓
detect_theme(text)  →  finds theme (intelligence / opinion / voice / etc.)
        ↓
generate_lora(text)                    generate_qlora(text)
  ↓ BART tokenizer                       ↓ T5 tokenizer + task prefix
  ↓ BART model inference                 ↓ T5 model inference
  ↓ rewrite_positive(text, "lora")       ↓ rewrite_positive(text, "qlora")
  ↓ picks empathetic message             ↓ picks motivational message
        ↓                                       ↓
   LoRA Output                            QLoRA Output
        ↓
detect_personality(text)  →  finds personality profile
        ↓
render_instagram_card(lora_output)
render_twitter_card(qlora_output)
        ↓
Display in Streamlit UI
```

### Example

```
Input  : "You are so dumb, nobody cares about your stupid opinion!"

Theme  : intelligence (contains "dumb", "stupid")

LoRA   : "Your unique perspective makes conversations richer! 🧠"
         (empathetic, community-focused)

QLoRA  : "Your brain sees things others miss — that's a superpower! ⚡"
         (motivational, energetic)

Personality : 😤 Frustrated Overachiever
              "Probably a stressed student who sets very high standards"
```

---

## 📊 Dataset Information

### Is a Dataset Used Directly in This Project?
**No.** This project performs **inference only** — using pre-trained models. No dataset is loaded or required at runtime.

### What Dataset Trained the Models?
Both `s-nlp/bart-base-detox` and `erfansadraiye/detoxify` were trained on the **ParaDetox** dataset by their respective authors before being uploaded to HuggingFace.

| Dataset | Size | Content | Used By |
|---------|------|---------|---------|
| **ParaDetox** | 19,000 pairs | Toxic sentence → Clean sentence pairs | s-nlp, erfansadraiye |

### Why No Dataset in This Project?
```
Training stage  (done by researchers, needs GPU + hours)
      ↓ uses ParaDetox dataset
Pre-trained detox model uploaded to HuggingFace
      ↓ you download it
Your project (inference only — no dataset needed) ✅
```

This is the same as using Google Translate — you don't need the translation dataset, because Google already trained the model on it.

---

## 🧰 Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11 |
| Web Framework | Streamlit 1.34.0 |
| Deep Learning | PyTorch 2.2.0 |
| NLP Models | HuggingFace Transformers 4.40.0 |
| Fine-Tuning | PEFT (Parameter Efficient Fine Tuning) 0.10.0 |
| LoRA Model | BART-base (s-nlp/bart-base-detox) |
| QLoRA Model | T5-Small (erfansadraiye/detoxify) |
| Data Handling | Pandas 2.2.2 |
| Visualization | Matplotlib 3.8.4 |
| Platform | Windows CPU (no GPU required) |

---

## 📋 Requirements

### Hardware
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB | 8GB+ |
| Storage | 3GB free | 5GB free |
| GPU | Not required | Not required |
| CPU | Any modern CPU | Any modern CPU |

### Software
| Software | Version |
|----------|---------|
| Windows | 10 or 11 |
| Python | 3.11 |
| pip | Latest |

---

## 🐛 Common Errors & Fixes

### Error 1 — `ImportError: cannot import name 'AutoTokenizer'`
**Cause:** transformers version is too old or corrupted.
```powershell
pip uninstall transformers -y
pip install transformers==4.40.0
```

### Error 2 — `BartForConditionalGeneration requires PyTorch`
**Cause:** PyTorch was uninstalled accidentally when upgrading transformers.
```powershell
pip uninstall torch -y
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.40.0
```

### Error 3 — `Using bitsandbytes 8-bit quantization requires GPU`
**Cause:** Tried to use `load_in_8bit=True` on CPU.
**Fix:** Use `low_cpu_mem_usage=True` instead — this is the correct CPU approach already in the project.

### Error 4 — `setup.bat` not recognized in PowerShell
**Cause:** PowerShell needs `.\` prefix.
```powershell
.\setup.bat   # correct
setup.bat     # wrong in PowerShell
```

### Error 5 — `matplotlib==0.8.4` install error
**Cause:** Typo in requirements — correct version is `3.8.4` not `0.8.4`.
```powershell
pip install matplotlib==3.8.4
```

### Error 6 — Models downloading every time app restarts
**Cause:** `@st.cache_resource` decorator missing.
**Fix:** Already handled in app.py — models load only once per session.

---



