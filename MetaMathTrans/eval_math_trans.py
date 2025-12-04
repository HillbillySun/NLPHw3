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
    if s is None:
        return None
    s = s.strip()
    left = "\\boxed{"
    # 只有在真的以 \boxed{ 开头、以 } 结尾时才去壳
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return s

def process_results(doc, completion, answer):
    # 1. 先抽预测表达式（已经是“去掉 The answer is:”的那部分）
    pred_expr = extract_answer_expr(completion)
    if pred_expr is None:
        invalid_outputs.append(
            {"question": doc, "output": completion, "answer": answer}
        )
        return False

    # 2. 去掉预测中的 \boxed{ }（如果有的话），但不要把普通表达式变 None
    pred_expr = remove_boxed(pred_expr)
    gold_expr = answer  # 这里 answer 已经是去壳后的纯表达式，比如 "0"、"\pi"

    # 3. 若任意一边还是 None，就直接判错并记录
    if pred_expr is None or gold_expr is None:
        invalid_outputs.append(
            {"question": doc, "output": completion, "answer": answer}
        )
        return False

    # 4. 用 util.is_equiv 做等价判断
    res = util.is_equiv(pred_expr, gold_expr)

    if not res:
        invalid_outputs.append(
            {"question": doc, "output": completion, "answer": answer}
        )
    # print("PRED:", pred_expr, "GOLD:", gold_expr)

    return res


# def process_results(doc, completion, answer):
    # # 1. 先用新函数抽预测表达式
    # pred_expr = extract_answer_expr(completion)
    # if pred_expr is None:
    #     invalid_outputs.append(
    #         {"question": doc, "output": completion, "answer": answer}
    #     )
    #     return False

    # # 2. 去掉 boxed 符号，和官方 eval 保持一致
    # pred_expr = remove_boxed(pred_expr)
    # gold_expr = remove_boxed(answer)

    # # 3. 用 util.is_equiv 做等价判断
    # res = util.is_equiv(pred_expr, gold_expr)

    # if not res:
    #     invalid_outputs.append(
    #         {"question": doc, "output": completion, "answer": answer}
    #     )

    # return res


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

# def extract_answer_expr(completion: str):
#     """
#     从模型输出中提取 'The answer is: ...' 后面的表达式。
#     没有这句时，退而求其次用最后一行非空文本。
#     """
#     # 1. 先尽量找 'The answer is'
#     m = re.search(r"[Tt]he answer is[:：]?\s*(.*)", completion)
#     if m:
#         expr = m.group(1).strip()
#         # 只取这一行，防止后面还有别的话
#         expr = expr.split("\n")[0].strip()
#     else:
#         # 2. 兜底：没有标记行，就拿最后一行非空内容
#         lines = [l.strip() for l in completion.splitlines() if l.strip()]
#         if not lines:
#             return None
#         expr = lines[-1]

#     # 去掉结尾的句号/感叹号之类
#     expr = expr.rstrip(".!；;")
#     return expr

def extract_answer_expr(completion: str):
    """
    先尽量从“真正的答案行”里抽:
        The answer is: <something>
    - 忽略模板里的 'The answer is: <expression>'
    - 没有的话，退到最后一个 \\boxed{...}
    - 再不行，用最后一行兜底
    """

    lines = completion.splitlines()

    # 1) 从“最后一行往前”找含 "The answer is" 的行
    for line in reversed(lines):
        m = re.search(r"[Tt]he answer is[:：]?\s*(.*)", line)
        if not m:
            continue

        expr = m.group(1).strip()
        # ---- 关键过滤：跳过模板那种行 ----
        # 例如 "The answer is: <expression>"
        if (
            not expr  # 空
            or "<" in expr or ">" in expr  # 含占位符 <>
            or "expression" in expr.lower()  # 明显是提示词里的字样
        ):
            continue

        expr = expr.rstrip(".!；;")
        return expr

    # 2) 退而求其次：找最后一个 \boxed{...}，沿用官方 MATH 的套路
    try:
        boxed = util.last_boxed_only_string(completion)
    except Exception:
        boxed = None

    if boxed is not None:
        expr = remove_boxed(boxed)  # "\\boxed{...}" -> "..."
        if expr is not None:
            expr = expr.strip().rstrip(".!；;")
            if expr:
                return expr

    # 3) 再兜底：拿最后一行非空内容，去掉常见前缀
    non_empty = [l.strip() for l in lines if l.strip()]
    if not non_empty:
        return None

    expr = non_empty[-1]
    expr = re.sub(
        r"^(Answer|So|Thus|Therefore|Final answer|The final answer is)[:：]?\s*",
        "",
        expr,
        flags=re.IGNORECASE,
    )
    expr = expr.rstrip(".!；;")
    return expr or None

def test_hendrycks_math(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    # problem_prompt = (
    #     "Below is an instruction that describes a task. "
    #     "Write a response that appropriately completes the request.\n\n"
    #     "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    # )

    # problem_prompt = (
    # "You are a helpful competition math assistant.\n"
    # "Below is a math problem. Solve it.\n"
    # "First, think step by step. Then on the last line, write "
    # "'The answer is: <expression>'.\n\n"
    # "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    # )
    problem_prompt = (
    "You are a helpful competition math assistant.\n"
    "Below is a math problem. Solve it.\n"
    "First, reason step by step.\n"
    "Then, on a NEW line at the end, output the final answer in the EXACT format:\n"
    "The answer is: <expression>\n"
    "Do NOT add any extra words after this line.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\nLet's think step by step."
    )


    print('promt =====', problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["instruction"])
            hendrycks_math_ins.append(temp_instr)
            solution = item['output']

            # 官方 MATH 写法：先取最后一个 boxed，然后去壳
            boxed = util.last_boxed_only_string(solution)   # 可能是 "\boxed{0}"
            temp_ans = remove_boxed(boxed)                  # 变成 "0"
            hendrycks_math_answers.append(temp_ans)


    print('total length ===', len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins[start:end]
    hendrycks_math_answers = hendrycks_math_answers[start:end]
    print('lenght ====', len(hendrycks_math_ins))

    # 初始化 HF 模型
    device = get_device()
    print("Using device:", device)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # GPU/MPS 用 float16，CPU 用 float32
    dtype = torch.float16 if device != "cpu" else torch.float32

    hf_model = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    hf_model.to(device)
    hf_model.eval()

    res_completions = []

    total = len(hendrycks_math_ins)
    bs = batch_size  # 直接用函数参数里的 batch_size

    for start_idx in tqdm(range(0, total, bs), desc="MATH eval", ncols=80):
        end_idx = min(start_idx + bs, total)
        batch_prompts = hendrycks_math_ins[start_idx:end_idx]

        # 批量编码，自动 padding
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # 每条样本真实长度（非 padding 的 token 数），用于截掉 prompt 部分
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = hf_model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 对 batch 里每一条分别解码
        for out_ids, in_len in zip(outputs, input_lengths):
            gen_ids = out_ids[in_len:]  # 截掉 prompt 部分，只保留新生成的 token
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            res_completions.append(text)



    results = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        res = process_results(prompt, completion, prompt_answer)
        results.append(res)

    acc = sum(results) / len(results)
    # print('len invalid outputs ====', len(invalid_outputs), ', invalid_outputs===', invalid_outputs)
    print('len invalid outputs ====', len(invalid_outputs))
    print('start===', start, ', end====',end)
    print('length====', len(results), ', acc====', acc)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=400)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=8)  # tensor_parallel_size
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_hendrycks_math(model=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size)
