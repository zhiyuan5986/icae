# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import json
import os
from collections import defaultdict
import torch

import numpy as np
from metrics import (
    classification_score,
    code_sim_score,
    count_score,
    qa_f1_score,
    qa_f1_zh_score,
    retrieval_score,
    retrieval_zh_score,
    rouge_score,
    rouge_zh_score,
)
from tqdm import tqdm
# from model.model import load_model_and_tokenizer, query_llm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, HfArgumentParser
from peft import LoraConfig
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from datasets import load_dataset
# from util.util import load_model_and_tokenizer, query_llm

# def query_llm(
#     prompt,
#     model,
#     model_name,
#     max_tokens,
#     tokenizer=None,
#     chat_completion=False,
#     **kwargs,
# ):

#     # 设置生成参数
#     generation_kwargs = {
#         "max_new_tokens": max_tokens,  # 设置最大长度
#         "eos_token_id": tokenizer.eos_token_id,  # 使用默认的结束标记
#         "pad_token_id": tokenizer.pad_token_id,  # 使用默认的填充标记
#         "early_stopping": True,  # 遇到结束标记时提前停止
#         # "no_repeat_ngram_si
#         # ze": 2,  # 避免重复的二元组
#         # "temperature": kwargs["temperature"] if "temperature" in kwargs else 0.7,
#         "top_p": kwargs["top_p"] if "top_p" in kwargs else 1.0,
#         "do_sample": False,
#     }
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     # print(inputs)
#     output = model.generate(**inputs, **generation_kwargs)
#     input_length = len(inputs['input_ids'][0]) if inputs['input_ids'][0].size(0) > 0 else 0
#     try:
#         answer = tokenizer.decode(output[0][input_length:], skip_special_tokens=True).strip()
#     except Exception as e:
#         print(f"Error decoding output: {e}")
#         answer = ""
#     return answer

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}
dataset2question = {
    'multi_news': 'You are given several news passages. Write a one-page summary of all news.',
    'gov_report': 'Write a one-page summary of the report.',
    'lcc': 'What is the next line for the code given below?',
    'passage_count': 'How many unique paragraphs there are after removing duplicated paragraphs?',
    'passage_count': 'Does this sentence contains meaningful information?',

    'lcc': 'What is the next line of code?',
    'repobench-p': 'What is the next line of code?',
}
from dataclasses import dataclass
@dataclass
class CustomArguments:
    n_max_token: int = 8100
    # load_prompt_from: str = "results/longbench/origin/longbench_test_single_doc_qa_formated.json"
    load_origin_from: str = "results/longbench/origin/longbench_test_single_doc_qa_formated.json"
    datasets: str = 'all'
    load_key: str = "prompt"
    target_tokens: int = 2000
    save_path: str = "results/longbench/origin/gpt35_chat_answer/answer_longbench_test_single_doc_qa_formated.json"
    e: bool = True
# parser = argparse.ArgumentParser(description="compress any prompt.")
# # parser.add_argument(
# #     "--model_name_or_path", help="LLM used to answer", default="gpt-3.5-turbo-0613"
# # )
# # parser.add_argument(
# #     "--tokenizer_name_or_path", help="tokenizer used to answer", default="gpt-3.5-turbo-0613"
# # )

# parser.add_argument("--n_max_token", type=int, default=8100)
# # parser.add_argument('--n_max_token_ans', type=int, default=400, help='token num in answer, following llmlingua')

# parser.add_argument(
#     "--load_prompt_from",
#     help="where to load compressed prompt",
#     default="results/longbench/origin/longbench_test_single_doc_qa_formated.json",
# )
# parser.add_argument("--load_key", default="prompt", type=str)
# parser.add_argument(
#     "--save_path",
#     help="path to save results",
#     default="results/longbench/origin/gpt35_chat_answer/answer_longbench_test_single_doc_qa_formated.json",
# )

# parser.add_argument("--e", action=argparse.BooleanOptionalAction, default=True)
# args = parser.parse_args()
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, CustomArguments))
model_args, data_args, training_args, args = parser.parse_args_into_dataclasses()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
all_datasets = [
    "narrativeqa", 
    "qasper", 
    "multifieldqa_en", 
    
    "hotpotqa", 
    "2wikimqa", 
    "musique", 
    
    "gov_report", 
    "qmsum", 
    "multi_news",

    'lcc',
    'repobench-p',

    'passage_count',
    'passage_retrieval_en',

    'trec',
    'triviaqa',
    'samsum',
]
# if args.e:
#     all_datasets = [
#         "qasper_e", 
#         "multifieldqa_en_e", 
        
#         "hotpotqa_e", 
#         "2wikimqa_e", 
        
#         "gov_report_e",  
#         "multi_news_e",

#         'lcc_e',
#         'repobench-p_e',

#         'passage_count_e',
#         'passage_retrieval_en_e',

#         'trec_e',
#         'triviaqa_e',
#         'samsum_e',
#     ]
datasets = all_datasets
if args.datasets != 'all':
    datasets = args.datasets.split(',')
dataset2question = {
    'multi_news': 'You are given several news passages. Write a one-page summary of all news.',
    'gov_report': 'Write a one-page summary of the report.',
    'lcc': 'What is the next line for the code given below?',
    'passage_count': 'How many unique paragraphs there are after removing duplicated paragraphs?',
    'passage_count': 'Does this sentence contains meaningful information?',

    'lcc': 'What is the next line of code?',
    'repobench-p': 'What is the next line of code?',
}
samples = []
for dataset_name in tqdm(datasets):
    if 'zh' in dataset_name or dataset_name in ['lsht']:
        continue
    # dataset = load_dataset('THUDM/LongBench', dataset_name, split='test')
    dataset = None
    if args.e:
        if os.path.exists(f'{args.load_origin_from}/{dataset_name}_e.jsonl'):
            dataset = load_dataset(path = 'json', data_files=f'{args.load_origin_from}/{dataset_name}_e.jsonl', split = 'train')
    else:
        dataset = load_dataset(path = 'json', data_files=f'{args.load_origin_from}/{dataset_name}.jsonl', split = 'train')
    if dataset:
        for d in dataset:
            d['input_is_null'] = (not d['input'] or d['input'][0] is None or d['input'][0].strip() == '')
            if (not d['input'] or d['input'][0] is None or d['input'][0].strip() == '') and dataset_name not in dataset2question:
                continue
            d['task'] = dataset_name
            d['idx'] = len(samples)
            d['question'] = dataset2question.get(dataset_name, d['input'])
            samples.append(d)



def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "multifieldqa_zh",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "dureader",
            "vcsum",
        ]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        # if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
        #     prediction = prediction.lstrip('\n').split('\n')[0]
        # for ground_truth in ground_truths:
        #     score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        # prediction = prediction.lstrip('\n').split('\n')[0]
        # prediction = prediction.strip("</s>")
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def eval(load_path):
    results = json.load(open(load_path))
    predictions, answers, lengths = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    all_classes = {}
    for idx, data in results.items():
        predictions[data["task"]].append(data["pred"])
        answers[data["task"]].append(data["answers"])
        all_classes[data["task"]] = data["all_classes"]
        if "length" in data:
            lengths[data["task"]].append(data["length"])
    scores = {}
    for task in predictions.keys():
        pred_list, ans_list, length_list = (
            predictions[task],
            answers[task],
            lengths[task],
        )
        if args.e:
            score = scorer_e(task, pred_list, ans_list, length_list, all_classes[task])
        else:
            score = scorer(task, pred_list, ans_list, all_classes[task])
        print(score)
        scores[task] = {"score": score, "num": len(pred_list)}
    score_list = [s["score"] for s in scores.values()]
    scores["avg"] = sum(score_list) / len(score_list)
    return scores


dataset2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    "passage_retrieval_zh": '以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n答案是：',
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}
dataset2instruction = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: ",
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: ',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n",
    "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n',
    "passage_retrieval_zh": '以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n',
    "lcc": "Please complete the code given below. \n",
    "repobench-p": "Please complete the code given below. \n",

    "pubmed": "You are given some medical passages. Write a one-page summary of these passages.\n\nPassages:\n",
    "meetingbank": "You are given meeting transcript. Write a one-page summary of this transcript.\n\nTranscript:\n",
    "summ_screen": "You are given several tv shows episodes. Write a one-page summary of these episodes.\n\nTranscript:\n",
}
dataset2questiontemplate = {
    "narrativeqa": "\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": '\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": "\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "\n\n会议总结：",
    "trec": "\n{input}",
    "triviaqa": "\n\n{input}",
    "samsum": "\n\n{input}",
    "lsht": "\n{input}",
    "passage_count": "\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": '\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    "passage_retrieval_zh": '\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n答案是：',
    "lcc": "Next line of code:\n",
    "repobench-p": "Next line of code:\n",

    "pubmed": "\n\nNow, write a one-page summary of the passages.\n\nSummary:",
    "meetingbank": "\n\nNow, write a one-page summary of the transcript.\n\nSummary:",
    "summ_screen": "\n\nNow, write a one-page summary of the episodes.\n\nSummary:",
}

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}


def predict():
    
    # Define Lora configuration
    lora_config = LoraConfig(
        r=128,
        lora_alpha=32,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Initialize model on CPU
    model = ICAE(model_args, training_args, lora_config)
    device = 'cuda'

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

    model = load_model_with_adjustments(model, training_args.output_dir)

    # Move the model to GPU
    # model = model.to(device)
    model.eval()
    # os.system("nvidia-smi | grep python")
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map = 'auto')
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path).to("cuda:2")
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    # dataset = json.load(open(args.load_prompt_from))
    dataset = samples
    print(len(dataset))
    if isinstance(dataset, dict):
        dataset = dataset.values()
    # dataset2prompt = json.load(
    #     open("../data/LongBench/config/dataset2prompt.json", "r")
    # )
    # dataset2maxlen = json.load(
    #     open("../data/LongBench/config/dataset2maxlen.json", "r")
    # )
    # prompt_format = dataset2prompt[args.task]
    # max_gen = int(dataset2maxlen[args.task])

    results = {}
    if os.path.exists(args.save_path):
        results = json.load(open(args.save_path))

    for sample in tqdm(dataset):
        # torch.cuda.empty_cache()
        idx = int(sample["idx"])
        # print("idx:", idx)
        # os.system("nvidia-smi")
        task = sample["task"]
        if idx in results or str(idx) in results:
            print(f"{idx} processed")
            continue
        new_sample = {}
        new_sample["context"] = sample[args.load_key]
        new_sample["input"] = sample["input"]
        new_sample["question"] = sample["question"]

        # prompt_format = dataset2prompt[sample["task"]]
        max_gen = int(dataset2maxlen[sample["task"]])
        # prompt = prompt_format.format(**new_sample)
        # token_ids = tokenizer.encode(prompt)
        
        prompt_left = "<s>▁[INST]" 
        prompt_right = new_sample["question"]
        tokenized_input = model.tokenizer(new_sample['context'], truncation=True, max_length=training_args.model_max_length, padding=False, return_attention_mask=False)
        # tokenized_input = model.tokenizer(new_sample['context'], padding=False, return_attention_mask=False)
        input_ids = torch.LongTensor([tokenized_input['input_ids']]).to(device)
        memory_slots = model._compress_target(input_ids, is_training=False, target_tokens=args.target_tokens)
        prompt_left_ids = model.tokenizer(prompt_left, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        prompt_right_ids = torch.cat([torch.LongTensor([[model.ft_token_id]]), model.tokenizer(prompt_right, add_special_tokens=False, return_tensors="pt").input_ids, model.tokenizer("[/INST]", add_special_tokens=False, return_tensors="pt").input_ids], dim=1).to(device)
        
        prompt_left_embs = model.tokens_to_embeddings(prompt_left_ids)
        prompt_right_embs = model.tokens_to_embeddings(prompt_right_ids)
        memory_slots = memory_slots.to(prompt_right_embs)
        print(input_ids.shape)
        print(memory_slots.shape)
                    
        # Concatenate and clone input embeddings
        decoder_input_embeddings = torch.cat((prompt_left_embs, memory_slots.unsqueeze(0), prompt_right_embs), dim=1)
        output = decoder_input_embeddings.clone()

        # generate_text = []
        # past_key_values = None
        # # Generate text output
        # for i in range(dataset2maxlen[task]):
        #     with model.icae.disable_adapter():   # no independent decoder; use self.icae
        #         out = model.icae(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
        #     logit = out.logits[:, -1, :model.vocab_size-1]
        #     past_key_values = out.past_key_values

        #     next_token_id = torch.argmax(logit, dim=-1)
            
        #     if next_token_id.item() == 2:   # eos
        #         break

        #     output = model.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).to(device)
        #     generate_text.append(next_token_id.item())
        # os.system("nvidia-smi")
        # os.system("nvidia-smi | grep python")

        generate_text = []
        past_key_values = None
        # Generate text output
        for i in range(dataset2maxlen[task]):
            # os.system("nvidia-smi | grep python")
            with model.icae.disable_adapter():   # no independent decoder; use self.icae
                with torch.no_grad():
                    out = model.icae(inputs_embeds=output, use_cache=True, past_key_values=past_key_values)
            logit = out.logits[:, -1, :model.vocab_size-1]
            # os.system("nvidia-smi | grep python")
            # os.system("nvidia-smi | grep python")
            past_key_values = out.past_key_values

            next_token_id = torch.argmax(logit, dim=-1)
            # print(next_token_id)
            
            if next_token_id.item() == 2:   # eos
                break
            # print(output.shape)
            # print(model.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).shape)
            # output = torch.cat([output, model.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).to(device)], dim=1)
            output = model.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).to(device)
            generate_text.append(next_token_id.item())

        # pred = model.tokenizer.decode(generate_text)

        pred = model.tokenizer.decode(generate_text).strip().split("\n\n")[0].split("<s>")[0].strip()

        # with model.icae.disable_adapter():
        #     out = model.icae.generate(inputs_embeds=decoder_input_embeddings, max_new_tokens=max_gen, use_cache=True, eos_token_id=2)
        # print(out)
        # pred = model.tokenizer.decode(out[0], skip_special_tokens=True).strip().split("\n\n")[0].split("<s>")[0].strip()

        # if len(token_ids) > (args.n_max_token - max_gen):
        #     half = int((args.n_max_token - max_gen) / 2) - 1
        #     prompt = tokenizer.decode(token_ids[:half]) + tokenizer.decode(
        #         token_ids[-half:]
        #     )

        # pred = query_llm(
        #     prompt, model, args.model_name_or_path, max_gen, tokenizer=tokenizer
        # )
        results[idx] = {
            "pred": pred,
            "answers": sample["answers"],
            "model_name": model_args.model_name_or_path,
            "task": sample["task"],
            "idx": idx,
            "all_classes": sample["all_classes"],
            "length": sample["length"],
        }
        json.dump(
            results,
            open(args.save_path, "w", encoding="utf8"),
            indent=4,
            ensure_ascii=False,
        )


# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3' 
predict()
score_dict = eval(load_path=args.save_path)
print(score_dict)
json.dump(
    score_dict,
    open(
        os.path.join(
            os.path.dirname(args.save_path),
            os.path.basename(args.save_path).replace("answer", "metrics"),
        ),
        "w",
    ),
)