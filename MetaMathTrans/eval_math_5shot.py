import argparse
import json
import pdb
import jsonlines
import re
import util
# from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
import sys
MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

invalid_outputs = []
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def process_results(doc, completion, answer):
    # 1. 先用新函数抽预测表达式
    pred_expr = extract_answer_expr(completion)
    if pred_expr is None:
        invalid_outputs.append(
            {"question": doc, "output": completion, "answer": answer}
        )
        return False

    # 2. 去掉 boxed 符号，和官方 eval 保持一致
    pred_expr = remove_boxed(pred_expr)
    gold_expr = remove_boxed(answer)

    # 3. 用 util.is_equiv 做等价判断
    res = util.is_equiv(pred_expr, gold_expr)

    if not res:
        invalid_outputs.append(
            {"question": doc, "output": completion, "answer": answer}
        )

    return res


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

def extract_answer_expr(completion: str):
    """
    从模型输出中提取 'The answer is: ...' 后面的表达式。
    没有这句时，退而求其次用最后一行非空文本。
    """
    # 1. 先尽量找 'The answer is'
    m = re.search(r"[Tt]he answer is[:：]?\s*(.*)", completion)
    if m:
        expr = m.group(1).strip()
        # 只取这一行，防止后面还有别的话
        expr = expr.split("\n")[0].strip()
    else:
        # 2. 兜底：没有标记行，就拿最后一行非空内容
        lines = [l.strip() for l in completion.splitlines() if l.strip()]
        if not lines:
            return None
        expr = lines[-1]

    # 去掉结尾的句号/感叹号之类
    expr = expr.rstrip(".!；;")
    return expr


def test_hendrycks_math(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    hendrycks_math_ins = []
    hendrycks_math_answers = []

    # 用前 K 条样本做 few-shot 示例
    with open(data_path, "r", encoding="utf8") as f:
        all_items = list(jsonlines.Reader(f))

    total_len = len(all_items)
    print("total length ===", total_len)

    # 防止样本太少的情况
    K = min(5, total_len)
    if K == 0:
        print("No data in", data_path)
        return

    fewshot_items = all_items[:K]
    eval_items = all_items[K:]  # 这些才参与评测

    # 单题模板（只描述当前要解的题）
    problem_prompt = (
        "### Problem:\n{instruction}\n\n"
        "### Solution:\nLet's think step by step."
    )
    print("problem template =====", problem_prompt)

    # -------- 1. 构造 few-shot 前缀（用 instruction / output 字段） --------
    fewshot_blocks = []
    for i, item in enumerate(fewshot_items, 1):
        # 题目在 instruction 字段
        q = item["instruction"]
        # 解答在 output 字段，里面包含 \boxed{...}
        solution = item["output"]

        # 从 gold solution 里抽出 \boxed{...} 里的表达式当答案
        # util.last_boxed_only_string 是你仓库里原来的工具函数
        ans_expr = remove_boxed(util.last_boxed_only_string(solution))

        block = (
            f"### Example {i} Problem:\n{q}\n\n"
            f"### Example {i} Solution:\n{solution}\n"
            f"The answer is: {ans_expr}\n\n"
        )
        fewshot_blocks.append(block)

    fewshot_prefix = (
        "You are a helpful competition math assistant.\n"
        "Below are some examples of solving competition math problems.\n"
        "For each problem, think step by step. Then on the last line, write "
        "'The answer is: <expression>'.\n\n"
        + "".join(fewshot_blocks)
        + "Now solve the following problem.\n\n"
    )

    # -------- 2. 为真正评测的题目构造输入 & gold 答案 --------
    for item in eval_items:
        # 这里用 instruction，不是 query
        temp_instr = fewshot_prefix + problem_prompt.format(
            instruction=item["instruction"]
        )
        hendrycks_math_ins.append(temp_instr)

        solution = item["output"]
        gold_expr = remove_boxed(util.last_boxed_only_string(solution))
        hendrycks_math_answers.append(gold_expr)

    print("eval total length ===", len(hendrycks_math_ins))

    # start / end 索引用在 eval 集上
    hendrycks_math_ins = hendrycks_math_ins[start:end]
    hendrycks_math_answers = hendrycks_math_answers[start:end]
    print("sliced length ====", len(hendrycks_math_ins))

    # -------- 3. 初始化 HF 模型（保持你原来的代码逻辑） --------
    device = get_device()
    print("Using device:", device)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    dtype = torch.float16 if device != "cpu" else torch.float32

    hf_model = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=True,
        dtype=dtype,
    )
    hf_model.to(device)
    hf_model.eval()

    res_completions = []

    total = len(hendrycks_math_ins)
    bs = batch_size

    for start_idx in tqdm(range(0, total, bs), desc="MATH eval", ncols=80):
        end_idx = min(start_idx + bs, total)
        batch_prompts = hendrycks_math_ins[start_idx:end_idx]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = hf_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for out_ids, in_len in zip(outputs, input_lengths):
            gen_ids = out_ids[in_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            res_completions.append(text)

    # -------- 4. 评测：用 extract_answer_expr + util.is_equiv --------
    results = []
    global invalid_outputs
    invalid_outputs = []

    for doc, completion, answer in zip(hendrycks_math_ins, res_completions, hendrycks_math_answers):
        pred_expr = extract_answer_expr(completion)
        if pred_expr is None:
            invalid_outputs.append(
                {"question": doc, "output": completion, "answer": answer}
            )
            results.append(False)
            continue

        pred_expr = remove_boxed(pred_expr)
        gold_expr = remove_boxed(answer)

        res = util.is_equiv(pred_expr, gold_expr)
        if not res:
            invalid_outputs.append(
                {"question": doc, "output": completion, "answer": answer}
            )
        results.append(res)

    acc = sum(results) / len(results) if results else 0.0
    print("len invalid outputs ====", len(invalid_outputs), ", invalid_outputs ===", invalid_outputs)
    print("length====", len(results), ", acc====", acc)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--train_file", type=str, required=True,  # 训练集路径
                        help="GSM8K 训练集 jsonl，用作 few-shot 示例")

    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=400)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=8)  # tensor_parallel_size
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_hendrycks_math(model=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size)
