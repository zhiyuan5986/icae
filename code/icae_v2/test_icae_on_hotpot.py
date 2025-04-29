import json
import torch
from tqdm import tqdm
from transformers import HfArgumentParser
from peft import LoraConfig
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from safetensors.torch import load_file
import string
import re
from collections import Counter

# Answer normalization function
def normalize_answer(s):
    """
    Normalize the answer by removing articles, punctuation, converting to lowercase, and fixing whitespace.

    Args:
        s (str): The answer string to be processed.

    Returns:
        str: The normalized answer string.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# F1 score calculation function
def compute_f1(prediction, truth):
    """
    Calculate the F1 score between the predicted answer and the true answer.

    Args:
        prediction (str): The answer predicted by the model.
        truth (str): The true answer.

    Returns:
        float: The F1 score.
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# Exact Match (EM) score calculation function
def compute_em(prediction, truth):
    """
    Calculate the Exact Match (EM) score between the predicted answer and the true answer.

    Args:
        prediction (str): The answer predicted by the model.
        truth (str): The true answer.

    Returns:
        int: The EM score, 1 for a match, 0 for no match.
    """
    return int(normalize_answer(prediction) == normalize_answer(truth))

# Set the computation device
device = "cuda"

# Parse model, data, training arguments and save_path
from dataclasses import dataclass
@dataclass
class CustomArguments:
    save_path: str = "./hotpot_dev_fullwiki_v1_prediction.json"

parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, CustomArguments))
model_args, data_args, training_args, custom_args = parser.parse_args_into_dataclasses()
save_path = custom_args.save_path

# Define Lora configuration
lora_config = LoraConfig(
    r=128,
    lora_alpha=32,
    lora_dropout=model_args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

# Initialize model and send it to CUDA device
model = ICAE(model_args, training_args, lora_config)

# Load the fine-tuned checkpoint
print(f"Loading trained checkpoint from {training_args.output_dir}")

def load_model_with_adjustments(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = model.state_dict()

    for name, param in checkpoint.items():
        if name in model_state_dict:
            if isinstance(param, torch.Tensor) and param.shape == model_state_dict[name].shape:
                model_state_dict[name] = param
            else:
                if not isinstance(param, torch.Tensor):
                    print(f"Skipping {name} because it's not a torch.Tensor (received {type(param)})")
                    # model_state_dict[name] = torch.Tensor(param)
                else:
                    print(f"Skipping {name} due to size mismatch. Checkpoint shape: {param.shape}, Model shape: {model_state_dict[name].shape}")
        else:
            print(f"Unexpected parameter {name} in checkpoint")

    model.load_state_dict(model_state_dict, strict=False)
    return model

# model.load_state_dict(torch.load(training_args.output_dir), strict=False)
model = load_model_with_adjustments(model, training_args.output_dir)

model = model.to(device)

# Read the data file
file_path = "/home/qiaoan/data/HotPotQA/hotpot_dev_fullwiki_v1.json"  # Adjust the path according to your actual file location
with open(file_path, "r") as f:
    data = json.load(f)

# Prepare the model for evaluation
max_out_length = 32
model.eval()

total_em = 0
total_f1 = 0
num_samples = len(data)

try:
    with open(save_path, "r") as f:
        existing_results = json.load(f)
    processed_ids = {result["_id"]: result for result in existing_results.get("samples", [])}
    summary = existing_results.get("summary", {"average_em_score": 0, "average_f1_score": 0})
    # total_em = summary["average_em_score"] * num_samples
    # total_f1 = summary["average_f1_score"] * num_samples
except FileNotFoundError:
    existing_results = {"samples": [], "summary": {}}
    processed_ids = {}
    summary = {"average_em_score": 0, "average_f1_score": 0}

with torch.no_grad():
    for sample in tqdm(data):
        sample_id = sample["_id"]
        # print(sample)
        context = " ".join([" ".join(para[1]) for para in sample.get('context', [])])
        question = sample["question"]
        answer = sample["answer"]

        input_text = context
        prompt_left = "<s>[INST]"
        instruction_w_question = "Using only the provided search results (some of which might be irrelevant), answer the following question with one or few words." + "\n\nQuestion: " + question + "\nAnswer: "

        if sample_id in processed_ids:
            result = processed_ids[sample_id]
            em_score = result["em_score"]
            f1_score = result["f1_score"]
        else:
            # Tokenize input text
            tokenized_input = model.tokenizer(input_text, truncation=True, max_length=5120, padding=False, return_attention_mask=False)
            # tokenized_prompt = model.tokenizer(prompt, truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)
            # Generate compressed outputs
            input_ids = torch.LongTensor([tokenized_input['input_ids']]).to(device)
            memory_slots = model._compress(input_ids)
            
            # decoder input has 3 parts: prefix, memory slots and suffix
            # the following code is for Mistral tokenizer for example: 733, 16289, 28793 are for the Mistral instruction tempmlate
            prompt_left_ids = model.tokenizer(prompt_left, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            # print(prompt_left_ids.shape)
            # prompt_left_ids = torch.LongTensor([prompt_left_ids]).to(device)
            # prompt_left_ids =  torch.LongTensor([[1, 733, 16289, 28793]]).to(device)
            # print(torch.LongTensor([model.ft_token_id]).shape)
            # print(model.tokenizer(instruction_w_question, add_special_tokens=False, return_tensors="pt").input_ids.shape)
            # print(model.tokenizer("[/INST]", add_special_tokens=False, return_tensors="pt").input_ids.shape)
            prompt_right_ids = torch.cat([torch.LongTensor([[model.ft_token_id]]), model.tokenizer(instruction_w_question, add_special_tokens=False, return_tensors="pt").input_ids, model.tokenizer("[/INST]", add_special_tokens=False, return_tensors="pt").input_ids], dim=1).to(device)
            # prompt_right_ids = [model.ft_token_id] + tokenized_prompt['input_ids'] + [733, 28748, 16289, 28793]
            # prompt_right_ids = torch.LongTensor([prompt_right_ids]).to(device)

            prompt_left_embs = model.tokens_to_embeddings(prompt_left_ids)
            prompt_right_embs = model.tokens_to_embeddings(prompt_right_ids)
            memory_slots = memory_slots.to(prompt_right_embs)
                        
            # Concatenate and clone input embeddings
            decoder_input_embeddings = torch.cat((prompt_left_embs, memory_slots.unsqueeze(0), prompt_right_embs), dim=1)
            output = decoder_input_embeddings.clone()

            generate_text = []
            past_key_values = None

            # Generate text output
            for i in range(max_out_length):
                with model.icae.disable_adapter():   # no independent decoder; use self.icae
                    out = model.icae(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
                logit = out.logits[:, -1, :model.vocab_size-1]
                past_key_values = out.past_key_values

                next_token_id = torch.argmax(logit, dim=-1)
                
                if next_token_id.item() == 2:   # eos
                    break

                output = model.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).to(device)
                generate_text.append(next_token_id.item())

            generated_text = model.tokenizer.decode(generate_text).strip().split("\n\n")[0].split("<s>")[0].strip()
            print("predicted_answer:", generated_text)
            print("answer:", answer)

            # Calculate EM and F1 scores
            em_score = compute_em(generated_text, answer)
            f1_score = compute_f1(generated_text, answer)

            # Structure output data
            result = {
                **sample,
                "predicted_answer": generated_text,
                "em_score": em_score,
                "f1_score": f1_score
            }
            existing_results["samples"].append(result)
            processed_ids[sample_id] = result

            # Dump results after each sample
            with open(save_path, "w") as f:
                json.dump(existing_results, f, indent=4)

        total_em += em_score
        total_f1 += f1_score

# Calculate average EM and F1 scores
avg_em = total_em / num_samples
avg_f1 = total_f1 / num_samples

# Update summary
summary = {
    "average_em_score": avg_em,
    "average_f1_score": avg_f1
}
existing_results["summary"] = summary

# Save final results with summary
with open(save_path, "w") as f:
    json.dump(existing_results, f, indent=4)

print(f"Results saved to {save_path}")
print(f"Average Exact Match (EM) Score: {avg_em * 100:.2f}%")
print(f"Average F1 Score: {avg_f1 * 100:.2f}%")