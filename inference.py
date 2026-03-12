"""
inference.py — Run the fine-tuned Disaster Response AI
=======================================================
Usage:
    python inference.py --method lora
    python inference.py --method qlora
    python inference.py --method qlora --scenario "Earthquake, 3 buildings collapsed"
    python inference.py --method qlora --interactive
"""

import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "EleutherAI/pythia-160m"
INSTRUCTION = "You are an expert disaster response coordinator. Provide a prioritized action plan for this emergency."

DEMO = [
    "7.1 earthquake. 3 buildings collapsed. 200 trapped. 4 rescue teams. Hospital 5km away.",
    "Flash flood. River burst banks. 2000 residents in 4 villages. Only 3 boats. Roads flooded.",
    "Chemical plant fire. Toxic black smoke. Wind toward residential area 300m away. 5 workers missing.",
]

def load(method):
    print(f"📥 Loading {method.upper()} model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    if method == "qlora":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.float32, bnb_4bit_use_double_quant=True)
        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb,
                                                     trust_remote_code=True, device_map="cpu")
        model = PeftModel.from_pretrained(base, f"results/{method}/adapter")
    else:
        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True)
        model = PeftModel.from_pretrained(base, f"results/{method}/adapter")

    model.eval()
    print("✅ Ready!\n")
    return model, tok

def respond(model, tok, scenario, max_new_tokens=200):
    prompt = f"### Instruction:\n{INSTRUCTION}\n\n### Scenario:\n{scenario}\n\n### Response:\n"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=400)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                              temperature=0.7, do_sample=True,
                              pad_token_id=tok.eos_token_id, repetition_penalty=1.2)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["lora", "qlora"], default="qlora")
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    model, tok = load(args.method)

    if args.interactive:
        print("🚨 Disaster Response AI | type 'quit' to exit\n")
        while True:
            s = input("📋 Scenario: ").strip()
            if s.lower() in ("quit", "q"): break
            if s: print(f"\n🔴 Response:\n{respond(model, tok, s)}\n")
    elif args.scenario:
        print(f"📋 {args.scenario}\n🔴 {respond(model, tok, args.scenario)}")
    else:
        for s in DEMO:
            print(f"📋 {s}\n🔴 {respond(model, tok, s)}\n{'─'*55}")