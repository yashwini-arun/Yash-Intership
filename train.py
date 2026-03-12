"""
train.py — Fine-tune pythia-160m using LoRA or QLoRA
=====================================================
Model  : EleutherAI/pythia-160m (160M params, ~320MB)
Device : CPU only

Usage:
    python train.py --method lora     # LoRA   (~28 min on CPU)
    python train.py --method qlora    # QLoRA  (~42 min on CPU)
"""

import os, time, argparse, json
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

MODEL_NAME = "EleutherAI/pythia-160m"

LORA_ARGS = dict(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def train(method: str):
    print(f"\n{'='*55}")
    print(f"  {'🔵 LoRA' if method=='lora' else '🟣 QLoRA'} — Disaster Response AI")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Device : CPU")
    print(f"{'='*55}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load Model
    print(f"\n📥 Loading model in {'4-bit NF4 (QLoRA)' if method=='qlora' else 'FP32 (LoRA)'}...")
    t0 = time.time()

    if method == "qlora":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb,
            trust_remote_code=True, device_map="cpu"
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, low_cpu_mem_usage=True
        )

    model.config.use_cache = False
    print(f"✅ Loaded in {time.time()-t0:.1f}s")

    # 3. Inject LoRA
    print("⚙️  Injecting LoRA adapters...")
    model = get_peft_model(model, LoraConfig(**LORA_ARGS))
    model.print_trainable_parameters()

    # 4. Dataset
    print("📦 Loading dataset...")
    train_data = read_jsonl("data/train.jsonl")
    test_data  = read_jsonl("data/test.jsonl")
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test":  Dataset.from_list(test_data),
    })
    print(f"   Train: {len(dataset['train'])} | Test: {len(dataset['test'])}")

    # 5. Tokenize
    def tokenize(examples):
        out = tokenizer(examples["text"], truncation=True,
                        max_length=512, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize, batched=True,
                            remove_columns=dataset["train"].column_names)

    # 6. Training Args
    output_dir = f"results/{method}"
    os.makedirs(output_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False, bf16=False,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        use_cpu=True,
        dataloader_num_workers=0,
    )

    # 7. Trainer
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
    )

    # 8. Train
    print(f"\n🚀 Training started... (~{'28' if method=='lora' else '42'} min on CPU)\n")
    t_start = time.time()
    trainer.train()
    print(f"\n✅ Done in {(time.time()-t_start)/60:.1f} min")

    # 9. Save
    adapter_path = f"results/{method}/adapter"
    os.makedirs(adapter_path, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"💾 Adapter saved → {adapter_path}")
    print(f"\n🎉 {method.upper()} fine-tuning complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["lora", "qlora"], required=True)
    args = parser.parse_args()
    train(args.method)