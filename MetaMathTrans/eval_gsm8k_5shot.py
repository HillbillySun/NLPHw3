import argparse
import json
import re
import jsonlines
from fraction import Fraction
# from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import random

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

# def extract_answer_number(completion):
#     text = completion.split('The answer is: ')
#     if len(text) > 1:
#         extract_ans = text[-1].strip()
#         match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
#         if match:
#             if '/' in match.group():
#                 denominator = match.group().split('/')[1]
#                 numerator = match.group().split('/')[0]
#                 if is_number(denominator) == True and is_number(numerator) == True:
#                     if denominator == '0':
#                         return round(float(numerator.replace(',', '')))
#                     else:
#                         frac = Fraction(match.group().replace(',', ''))
#                         num_numerator = frac.numerator
#                         num_denominator = frac.denominator
#                         return round(float(num_numerator / num_denominator))
#                 else:
#                     return None
#             else:
#                 if float(match.group().replace(',', '')) == float('inf'):
#                     return None
#                 return round(float(match.group().replace(',', '')))
#         else:
#             return None
#     else:
#         return None


def extract_answer_number(completion):
    """
    先尝试严格模式：在 'The answer is:' 后面抓一个数字；
    如果没抓到，再退而求其次：从整段输出里取最后一个数字。
    """

    def parse_one_number(s):
        """把一个字符串里的第一个数字解析成 int，支持分数/逗号。"""
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', s)
        if not match:
            return None

        token = match.group().replace(',', '')
        # 分数情况
        if '/' in token:
            num, den = token.split('/', 1)
            if not (is_number(num) and is_number(den)):
                return None
            if float(den) == 0:
                return round(float(num))
            frac = Fraction(token)
            return round(float(frac.numerator / frac.denominator))
        else:
            if not is_number(token):
                return None
            if float(token) == float('inf'):
                return None
            return round(float(token))

    # ---------- 严格模式：找 "The answer is:" ----------
    if 'The answer is' in completion:
        # 兼容 'The answer is:' / 'The answer is -' 这种
        parts = completion.split('The answer is', 1)
        tail = parts[1]
        # 去掉可能紧跟的冒号/空格
        tail = tail.lstrip(' :')
        val = parse_one_number(tail)
        if val is not None:
            return val

    # ---------- 宽松模式：抓整段里的最后一个数字 ----------
    matches = re.findall(r'[\-+]?\d*[\.,/]?\d+', completion)
    if not matches:
        return None

    last_token = matches[-1].replace(',', '')
    # 分数
    if '/' in last_token:
        num, den = last_token.split('/', 1)
        if not (is_number(num) and is_number(den)):
            return None
        if float(den) == 0:
            return round(float(num))
        frac = Fraction(last_token)
        return round(float(frac.numerator / frac.denominator))
    else:
        if not is_number(last_token):
            return None
        if float(last_token) == float('inf'):
            return None
        return round(float(last_token))

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

     # ---- 固定随机种子 ----
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    
    # 这段模板现在只用来打印
    problem_prompt = (
        "You are a helpful math assistant.\n"
        "Below is a math word problem. Solve it.\n"
        "First, think step by step. Then on the last line, write "
        "'The answer is: <number>'.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )

    K = 5  # few-shot 示例数量
    print('promt =====', problem_prompt)

    # ========= 1）读取 train_file，作为 few-shot 仓库 =========
    # 支持两种情况：
    #   - *.json  : MetaMathQA-1k.json 这类大 JSON
    #   - *.jsonl : GSM8K_train.jsonl 这类一行一条
    if train_path.endswith(".jsonl"):
        # 视作 GSM8K 风格 jsonl（字段：query / response）
        with open(train_path, "r", encoding="utf8") as f:
            train_items = list(jsonlines.Reader(f))

        def get_q(it):     # 取题面
            return it["query"]
        def get_resp(it):  # 取带 "####" 的完整解答
            return it["response"]
    else:
        # 视作 MetaMathQA 的大 json
        with open(train_path, "r", encoding="utf8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            train_items = raw
        elif isinstance(raw, dict):
            if isinstance(raw.get("train"), list):
                train_items = raw["train"]
            elif isinstance(raw.get("data"), list):
                train_items = raw["data"]
            else:
                train_items = None
                for v in raw.values():
                    if isinstance(v, list):
                        train_items = v
                        break
                if train_items is None:
                    raise ValueError("无法从 train_file json 中解析出样本列表，请检查文件结构。")
        else:
            raise ValueError("train_file 的顶层既不是 list 也不是 dict，请检查文件格式。")

        def get_q(it):
            # MetaMathQA 可能是 query / instruction 之一
            return it.get("query") or it.get("instruction")
        def get_resp(it):
            return it.get("response") or it.get("output")

    # ========= 2）在 train_items 中选出“看起来像 GSM”的样本 =========
    gsm_train_items = []
    for it in train_items:
        t = str(it.get("type", "")).upper()
        q_text = (get_q(it) or "").lower()
        # type 以 GSM 开头，且题面里不含 "unknown variable" 这类明显 meta 的句式
        # if t.startswith("GSM") and "unknown variable" not in q_text and "if we know the answer" not in q_text:
        #     gsm_train_items.append(it)

        if t in ("GSM_ANSAUG",):
            gsm_train_items.append(it)


    if len(gsm_train_items) == 0:
        print("[Warning] train_file 中没有合适的 GSM 样本，将退化为使用全部 train_items 做 few-shot。")
        gsm_train_items = train_items

    # few-shot 数量 K 的实际值
    if len(gsm_train_items) < K:
        print(f"[Warning] few-shot 只找到 {len(gsm_train_items)} 条样本，K 自动改为 {len(gsm_train_items)}")
        K_local = len(gsm_train_items)
    else:
        K_local = K

    # 随机采样 K 条，不要老是前 5 条（避免排序偏差）
    fewshot_items = random.sample(gsm_train_items, K_local)

    # ========= 3）读取测试集 =========
    with open(data_path, "r", encoding="utf8") as f:
        eval_items = list(jsonlines.Reader(f))

    # ========= 4）构造 few-shot 前缀 =========
    fewshot_blocks = []
    for i, item in enumerate(fewshot_items, 1):
        q = get_q(item)
        full_resp = (get_resp(item) or "").strip()

        # 尝试从 full_resp 中拆出【思路部分】和【最终数字答案】
        reasoning = full_resp
        ans_str = None

        if "The answer is:" in full_resp:
            # 优先用 "The answer is:" 这一行
            reason, after = full_resp.split("The answer is:", 1)
            reasoning = reason.strip()
            m = re.search(r'[\-+]?\d*[\.,/]?\d+', after)
            if m:
                ans_str = m.group(0).replace(",", "")
        elif "####" in full_resp:
            # GSM8K/MetaMath 常见：...... #### 18
            parts = full_resp.split("####")
            reasoning = parts[0].strip()
            tail = parts[-1].strip()
            m = re.search(r'[\-+]?\d*[\.,/]?\d+', tail)
            if m:
                ans_str = m.group(0).replace(",", "")

        # 如果没成功解析出数字答案，就跳过这条作为 few-shot
        if ans_str is None:
            continue

        block = (
            f"### Example {i} Instruction:\n{q}\n\n"
            f"### Example {i} Response:\n{reasoning}\n"
            f"The answer is: {ans_str}\n\n"
        )
        fewshot_blocks.append(block)

    fewshot_prefix = "".join(fewshot_blocks)

    # ========= 5）为每一道测试题构造带 few-shot 的 prompt =========
    for item in eval_items:
        temp_instr = (
            "You are a helpful math assistant.\n"
            "Below are some examples of solving math word problems.\n"
            "For each new problem, follow the same format:\n"
            "think step by step, and on the LAST line output exactly:\n"
            "'The answer is: <number>'.\n\n"
            f"{fewshot_prefix}"
            f"### Instruction:\n{item['query']}\n\n"
            "### Response: Let's think step by step."
        )
        gsm8k_ins.append(temp_instr)

        # 测试集：仍然假定是 GSM8K 样式的 "... #### <number>"
        temp_ans = item["response"].split("#### ")[1]
        temp_ans = int(temp_ans.replace(",", ""))
        gsm8k_answers.append(temp_ans)

    # ========= 6）裁 slice，便于只测一部分 =========
    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('length after slice ====', len(gsm8k_ins))

    # ========= 7）初始化 HF 模型 =========
    device = get_device()
    print("Using device:", device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left" 

    dtype = torch.float16 if device != "cpu" else torch.float32
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    hf_model.to(device)
    hf_model.eval()

    # ========= 8）批量生成 =========
    result = []
    res_completions = []

    total = len(gsm8k_ins)
    bs = batch_size

    for offset in tqdm(range(0, total, bs), desc="GSM8K eval", ncols=80):
        end_idx = min(offset + bs, total)
        batch_prompts = gsm8k_ins[offset:end_idx]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,  
        )

        input_lengths = (inputs["attention_mask"].sum(dim=1)).tolist()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = hf_model.generate(
                **inputs,
                max_new_tokens=256,          # 够 CoT + 最后一行用
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for out_ids, in_len in zip(outputs, input_lengths):
            gen_ids = out_ids[in_len:]  # 截掉 prompt 部分
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            res_completions.append(text)

    # ========= 9）解析答案 & 计算准确率 =========
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(
        zip(gsm8k_ins, res_completions, gsm8k_answers)
    ):
        y_pred = extract_answer_number(completion)
        if y_pred is not None:
            result.append(float(y_pred) == float(prompt_answer))
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)

    acc = sum(result) / len(result) if result else 0.0
    print('len invalid outputs ====', len(invalid_outputs), ', invalid_outputs (first 3)===', invalid_outputs[:3])
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
