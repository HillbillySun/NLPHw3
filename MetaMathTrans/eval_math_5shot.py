import argparse
import json
import jsonlines
import re
import util
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
    """只在确实是 \\boxed{...} 时去壳，否则原样返回。"""
    if s is None:
        return None
    s = s.strip()
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left):-1]
    return s

def extract_answer_expr(completion: str):
    """
    从模型输出中提取答案表达式（预测用），优先级：
    1) 真实答案行: The answer is: <expr>（跳过模板里的 <expression>）
    2) 最后一个 \\boxed{...}
    3) 最后一行非空内容兜底，并去掉常见前缀。
    """
    lines = completion.splitlines()

    # 1) 从末尾往前找含 "The answer is" 的行，跳过模板那种
    for line in reversed(lines):
        m = re.search(r"[Tt]he answer is[:：]?\s*(.*)", line)
        if not m:
            continue

        expr = m.group(1).strip()
        # 跳过模板里的占位：<expression> / 空行 / 包含 <> 等
        if (
            not expr
            or "<" in expr or ">" in expr
            or "expression" in expr.lower()
        ):
            continue

        expr = expr.rstrip(".!；;")
        if expr:
            return expr

    # 2) 退而求其次：找最后一个 \\boxed{...}
    try:
        boxed = util.last_boxed_only_string(completion)
    except Exception:
        boxed = None

    if boxed is not None:
        expr = remove_boxed(boxed)
        if expr is not None:
            expr = expr.strip().rstrip(".!；;")
            if expr:
                return expr

    # 3) 再兜底：取最后一行非空内容，并去掉常见前缀
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


def extract_gold_expr_from_solution(solution: str):
    """
    从 train/test 的标准解答文本中抽金答案表达式，优先级：
    1) The answer is: <expr>
    2) 最后一个 \\boxed{...}
    抽不到就返回 None。
    """
    if solution is None:
        return None

    # 1) 先找 "The answer is: ..."
    m = re.search(r"[Tt]he answer is[:：]?\s*(.*)", solution)
    if m:
        expr = m.group(1).strip()
        expr = expr.split("\n")[0].strip()
        expr = expr.rstrip(".!；;")
        expr = remove_boxed(expr)
        if expr:
            return expr

    # 2) 退到 \boxed{...}
    try:
        boxed = util.last_boxed_only_string(solution)
    except Exception:
        boxed = None

    if boxed is not None:
        expr = remove_boxed(boxed)
        if expr is not None:
            expr = expr.strip().rstrip(".!；;")
            if expr:
                return expr

    return None


def process_results(doc, completion, gold_expr):
    """
    对单个样本做评测：
    - 从 completion 中抽预测表达式
    - 与 gold_expr 用 util.is_equiv 判等价
    - 记录错误样本到 invalid_outputs
    """
    global invalid_outputs

    pred_expr = extract_answer_expr(completion)
    if pred_expr is None:
        invalid_outputs.append(
            {"question": doc, "output": completion, "answer": gold_expr}
        )
        print("PRED: None", "GOLD:", gold_expr)
        return False

    # 万一模型自己又包了个 \boxed{}，这里再去一层
    pred_expr = remove_boxed(pred_expr)

    if pred_expr is None or gold_expr is None:
        invalid_outputs.append(
            {"question": doc, "output": completion, "answer": gold_expr}
        )
        print("PRED:", pred_expr, "GOLD:", gold_expr)
        return False

    res = util.is_equiv(pred_expr, gold_expr)

    if not res:
        invalid_outputs.append(
            {"question": doc, "output": completion, "answer": gold_expr}
        )
    print("PRED:", pred_expr, "GOLD:", gold_expr)
    return res


def test_hendrycks_math(model, data_path, train_path,
                        start=0, end=MAX_INT, batch_size=1,
                        tensor_parallel_size=1):
    """
    MATH 5-shot 评测：
    - train_path: few-shot 示例来源（json 或 jsonl，都支持 MetaMathQA-1k.json）
    - data_path : 评测集（比如 MATH_test.jsonl）
      样本中尽量有：instruction/output 或 query/response 等字段。
    """
    global invalid_outputs
    invalid_outputs = []

    # ---------- 1. 读取 train_file，构造 few-shot 示例 ----------
    with open(train_path, "r", encoding="utf8") as f:
        if train_path.endswith(".jsonl"):
            train_items = list(jsonlines.Reader(f))
        else:
            raw = json.load(f)
            if isinstance(raw, list):
                train_items = raw
            elif isinstance(raw, dict):
                # 常见结构：{"train": [...]} / {"data": [...]}
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
                raise ValueError("train_file 既不是 list 也不是 dict，请检查格式。")

    # 尝试只保留 MATH 类型（针对 MetaMathQA 这类）
    filtered_items = []
    for it in train_items:
        t = str(it.get("type", "")).upper()
        if "MATH" in t:
            filtered_items.append(it)

    if filtered_items:
        train_items = filtered_items
        print(f"[info] 只使用 type 含 'MATH' 的 train 样本，数量 = {len(train_items)}")
    else:
        print("[warn] train_file 中没有带 'MATH' 类型字段，使用全部样本做 few-shot。")

    train_total = len(train_items)
    print("train total length ===", train_total)

    K = min(5, train_total)
    if K == 0:
        print("Empty train_file after filtering:", train_path)
        return

    fewshot_items = train_items[:K]  # 你也可以改成 random.sample(train_items, K)

    def get_train_instruction(it):
        return (
            it.get("instruction")
            or it.get("query")
            or it.get("problem")
            or it.get("question")
        )

    def get_train_output(it):
        return (
            it.get("output")
            or it.get("response")
            or it.get("solution")
        )

    # few-shot 示例块
    fewshot_blocks = []
    for i, item in enumerate(fewshot_items, 1):
        q = get_train_instruction(item)
        solution = get_train_output(item)

        if q is None or solution is None:
            # 跳过格式不完整的样本
            continue

        # 从标准解答里抽 gold expr
        ans_expr = extract_gold_expr_from_solution(solution)

        if ans_expr is None:
            # 没抽到答案，就只给原始解答，不额外加 'The answer is: None'
            block = (
                f"### Example {i} Problem:\n{q}\n\n"
                f"### Example {i} Solution:\n{solution}\n\n"
            )
        else:
            block = (
                f"### Example {i} Problem:\n{q}\n\n"
                f"### Example {i} Solution:\n{solution}\n"
                f"The answer is: {ans_expr}\n\n"
            )
        fewshot_blocks.append(block)

    if not fewshot_blocks:
        raise ValueError("few-shot 样本都不合法（instruction/output 缺失），请检查 train_file。")

    # few-shot 前缀
    fewshot_prefix = (
        "You are a helpful competition math assistant.\n"
        "Below are some examples of solving competition math problems.\n"
        "For each problem, reason step by step and on the last line output:\n"
        "The answer is: <expression>\n\n"
        + "".join(fewshot_blocks)
        + "Now solve the following problem.\n\n"
    )

    # 单题模板
    problem_prompt = (
        "### Problem:\n{instruction}\n\n"
        "### Solution:\nLet's think step by step."
    )
    print("single problem template =====", problem_prompt)

    # ---------- 2. 读取 test_file（data_path），构造真正的评测输入 ----------
    hendrycks_math_ins = []
    hendrycks_math_answers = []

    with open(data_path, "r", encoding="utf8") as f:
        if data_path.endswith(".jsonl"):
            all_items = list(jsonlines.Reader(f))
        else:
            raw = json.load(f)
            if isinstance(raw, list):
                all_items = raw
            elif isinstance(raw, dict):
                if isinstance(raw.get("data"), list):
                    all_items = raw["data"]
                else:
                    all_items = None
                    for v in raw.values():
                        if isinstance(v, list):
                            all_items = v
                            break
                    if all_items is None:
                        raise ValueError("无法从 data_file json 中解析出样本列表，请检查文件结构。")
            else:
                raise ValueError("data_file 既不是 list 也不是 dict，请检查格式。")

    print("test total length ===", len(all_items))

    def get_test_instruction(it):
        return (
            it.get("instruction")
            or it.get("query")
            or it.get("problem")
            or it.get("question")
        )

    def get_test_output(it):
        return (
            it.get("output")
            or it.get("response")
            or it.get("solution")
        )

    for item in all_items:
        instr = get_test_instruction(item)
        solution = get_test_output(item)
        if instr is None or solution is None:
            continue

        gold_expr = extract_gold_expr_from_solution(solution)
        if gold_expr is None:
            # gold 抽不到答案，没法评测，直接跳过
            continue

        full_prompt = fewshot_prefix + problem_prompt.format(instruction=instr)
        hendrycks_math_ins.append(full_prompt)
        hendrycks_math_answers.append(gold_expr)

    # 应用 start / end 截取
    hendrycks_math_ins = hendrycks_math_ins[start:end]
    hendrycks_math_answers = hendrycks_math_answers[start:end]
    print("eval sliced length ====", len(hendrycks_math_ins))

    if not hendrycks_math_ins:
        print("Nothing to evaluate after slicing.")
        return

    # ---------- 3. 初始化 HF 模型 ----------
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
        torch_dtype=dtype,  # 目前用 torch_dtype，避免部分模型报错
    )
    hf_model.to(device)
    hf_model.eval()

    # ---------- 4. 批量生成 ----------
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
                do_sample=False,               # 需要可以改 True + 设置 temperature/top_p
                pad_token_id=tokenizer.pad_token_id,
            )

        for out_ids, in_len in zip(outputs, input_lengths):
            gen_ids = out_ids[in_len:]  # 截掉 prompt
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            res_completions.append(text)

    # ---------- 5. 评测 ----------
    results = []
    for doc, completion, gold_expr in zip(
        hendrycks_math_ins, res_completions, hendrycks_math_answers
    ):
        res = process_results(doc, completion, gold_expr)
        results.append(res)

    acc = sum(results) / len(results) if results else 0.0
    print("len invalid outputs ====", len(invalid_outputs),
          ", invalid_outputs ===", invalid_outputs)
    print("length====", len(results), ", acc====", acc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='',
                        help="HF 模型名或本地路径，建议用 *-Instruct 模型")
    parser.add_argument("--data_file", type=str, default='',
                        help="测试集（例如 MATH_test.jsonl）")
    parser.add_argument("--train_file", type=str, required=True,
                        help="MATH 训练集 / few-shot 源（json 或 jsonl，支持 MetaMathQA-1k.json）")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=MAX_INT)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_hendrycks_math(
        model=args.model,
        data_path=args.data_file,
        train_path=args.train_file,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
    )
