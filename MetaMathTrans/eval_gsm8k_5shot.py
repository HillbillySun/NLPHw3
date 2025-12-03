import argparse
import json
import re
import jsonlines
from fraction import Fraction
# from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

import sys
MAX_INT = sys.maxsize

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def gsm8k_test(model_name, data_path, train_path,
               start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):

    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    
    problem_prompt = (
        "You are a helpful math assistant.\n"
        "Below is a math word problem. Solve it.\n"
        "First, think step by step. Then on the last line, write "
        "'The answer is: <number>'.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )

    K = 5
    print('promt =====', problem_prompt)

    # ========= 1）从 MetaMathQA 读 few-shot 示例 =========
    # 先读训练集，用作 few-shot 示例（MetaMathQA 是一个 JSON 文件，不是 jsonl）
    with open(train_path, "r", encoding="utf8") as f:
        raw = json.load(f)

    # 兼容几种常见格式：
    # 1. 直接是一个 list：[ {row1}, {row2}, ... ]
    # 2. 是一个 dict，里面某个 key（通常是 "train" 或 "data"）对应一个 list
    if isinstance(raw, list):
        train_items = raw
    elif isinstance(raw, dict):
        if isinstance(raw.get("train"), list):
            train_items = raw["train"]
        elif isinstance(raw.get("data"), list):
            train_items = raw["data"]
        else:
            # 尝试从第一个 list 类型的 value 里取
            train_items = None
            for v in raw.values():
                if isinstance(v, list):
                    train_items = v
                    break
            if train_items is None:
                raise ValueError("无法从 MetaMathQA json 中解析出样本列表，请检查文件结构。")
    else:
        raise ValueError("MetaMathQA json 文件的顶层既不是 list 也不是 dict，请检查文件格式。")


    # 只保留 GSM 类型样本（MetaMathQA: type 一般形如 "GSM_AnsAug", "GSM_FOBAR" 等）
    gsm_train_items = []
    for it in train_items:
        t = it.get("type", "")
        if isinstance(t, str) and t.upper().startswith("GSM"):
            gsm_train_items.append(it)

    if len(gsm_train_items) == 0:
        print("[Warning] train_file 中没有 GSM 类型样本，将退化为用全部 train_items 做 few-shot。")
        gsm_train_items = train_items

    if len(gsm_train_items) < K:
        print(f"[Warning] GSM few-shot 只找到 {len(gsm_train_items)} 条样本，K 自动改为 {len(gsm_train_items)}")
        K_local = len(gsm_train_items)
    else:
        K_local = K

    fewshot_items = gsm_train_items[:K_local]

    # ========= 2）读测试集（真实评测对象） =========
    with open(data_path, "r", encoding="utf8") as f:
        eval_items = list(jsonlines.Reader(f))

    # ========= 3）把 few-shot 示例展开成前缀 =========
    fewshot_blocks = []
    for i, item in enumerate(fewshot_items, 1):
        q = item["query"]                 # MetaMathQA 的题目
        full_resp = item["response"].strip()  # 完整 CoT + The answer is: ...

        # 不再用 "####" 切分，直接把官方 CoT 全贴进去
        block = (
            f"### Example {i} Instruction:\n{q}\n\n"
            f"### Example {i} Response:\n{full_resp}\n\n"
        )
        fewshot_blocks.append(block)

    fewshot_prefix = "".join(fewshot_blocks)

    # ========= 4）为每一道测试题构造带 few-shot 的 prompt =========
    for item in eval_items:
        temp_instr = (
            "You are a helpful math assistant.\n"
            "Below are some examples of solving math word problems.\n"
            "For each problem, think step by step, then on the last line write "
            "'The answer is: <number>'.\n\n"
            f"{fewshot_prefix}"
            f"### Instruction:\n{item['query']}\n\n"
            "### Response: Let's think step by step."
        )
        gsm8k_ins.append(temp_instr)

        # 测试集仍然假定是 MetaMath 原始 GSM8K 格式：... #### <number>
        temp_ans = item["response"].split("#### ")[1]
        temp_ans = int(temp_ans.replace(",", ""))
        gsm8k_answers.append(temp_ans)

    # 这里再裁一遍 start/end，方便只跑一部分
    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('length after slice ====', len(gsm8k_ins))

    # ========= 5）初始化 HF 模型 =========
    device = get_device()
    print("Using device:", device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # CPU 用 fp32，更稳；GPU/MPS 用 fp16 省显存
    dtype = torch.float16 if device != "cpu" else torch.float32
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    hf_model.to(device)
    hf_model.eval()

    result = []
    res_completions = []

    total = len(gsm8k_ins)
    bs = batch_size

    for offset in tqdm(range(0, total, bs), desc="GSM8K eval", ncols=80):
        end_idx = min(offset + bs, total)
        batch_prompts = gsm8k_ins[offset:end_idx]

        # 批量编码
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,   # 可调：限制 prompt 最大长度，防止显存爆
        )

        # 每条真实输入长度（用来截生成部分）
        input_lengths = (inputs["attention_mask"].sum(dim=1)).tolist()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = hf_model.generate(
                **inputs,
                max_new_tokens=256,          # 可调：生成长度，太长会拖慢/占显存
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for out_ids, in_len in zip(outputs, input_lengths):
            gen_ids = out_ids[in_len:]  # 截掉 prompt 部分
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            res_completions.append(text)

    # ========= 6）解析 & 计算 acc =========
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(
        zip(gsm8k_ins, res_completions, gsm8k_answers)
    ):
        doc = {'question': prompt}
        y_pred = extract_answer_number(completion)
        if y_pred is not None:
            result.append(float(y_pred) == float(prompt_answer))
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)

    acc = sum(result) / len(result) if result else 0.0
    print('len invalid outputs ====', len(invalid_outputs), ', invalid_outputs===', invalid_outputs[:3])
    print('start===', start, ', end====', end)
    print('gsm8k length====', len(result), ', gsm8k acc====', acc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--train_file", type=str, required=True,
                    help="GSM8K 训练集 jsonl，用作 few-shot 示例")
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=400)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=8)  # tensor_parallel_size
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    gsm8k_test(
        model_name=args.model,
        data_path=args.data_file,      # test
        train_path=args.train_file,    # train
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )
