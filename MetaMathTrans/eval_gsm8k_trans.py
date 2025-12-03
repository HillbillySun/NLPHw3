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


def gsm8k_test(model_name, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
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

    print('promt =====', problem_prompt)
    with open(data_path,"r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["query"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item['response'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    # gsm8k_ins = gsm8k_ins[start:end]
    # gsm8k_answers = gsm8k_answers[start:end]
    # print('lenght ====', len(gsm8k_ins))
    # batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    # stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
    # sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=512, stop=stop_tokens)
    # print('sampleing =====', sampling_params)
    # llm = LLM(model=model,tensor_parallel_size=tensor_parallel_size)
    # result = []
    # res_completions = []
    # for idx, (prompt, prompt_answer) in enumerate(zip(batch_gsm8k_ins, gsm8k_answers)):
    #     if isinstance(prompt, list):
    #         pass
    #     else:
    #         prompt = [prompt]

    #     completions = llm.generate(prompt, sampling_params)
    #     for output in completions:
    #         prompt = output.prompt
    #         generated_text = output.outputs[0].text
    #         res_completions.append(generated_text)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('lenght ====', len(gsm8k_ins))

    # 初始化 HF 模型
    device = get_device()
    print("Using device:", device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "left"

    if device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    hf_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name,
    trust_remote_code=True,
    torch_dtype=dtype,   # 关键：用 FP16
    # device_map="auto",           # 单卡时等价于全放到 cuda:0
)
    hf_model.to(device)
    hf_model.eval()

    result = []
    res_completions = []

    # for prompt in tqdm(gsm8k_ins, desc="GSM8K eval", ncols=80):
    #     # 单条生成（简单稳定，batch_size 参数可以无视）
    #     inputs = tokenizer(prompt, return_tensors="pt")
    #     input_len = inputs["input_ids"].shape[1]
    #     inputs = {k: v.to(device) for k, v in inputs.items()}

    #     with torch.no_grad():
    #         output_ids = hf_model.generate(
    #             **inputs,
    #             max_new_tokens=512,
    #             do_sample=False,
    #             temperature=0.0,
    #             pad_token_id=tokenizer.pad_token_id,
    #         )[0]

    #     # 只拿生成部分
    #     gen_ids = output_ids[input_len:]
    #     generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    #     res_completions.append(generated_text)
    total = len(gsm8k_ins)
    bs = batch_size  # argparse 解析出来的

    for start in tqdm(range(0, total, bs), desc="GSM8K eval", ncols=80):
        end = min(start + bs, total)
        batch_prompts = gsm8k_ins[start:end]
        # 1) 批量编码，自动 padding
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_lens = inputs["input_ids"].shape[1]  # 这里只是最大长度，下面会用到每条的长度

        # 真正的每条输入长度，用来截生成部分
        input_lengths = (inputs["attention_mask"].sum(dim=1)).tolist()

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = hf_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 2) 对 batch 里每一条分别解码
        for out_ids, in_len in zip(outputs, input_lengths):
            gen_ids = out_ids[in_len:]  # 截掉 prompt 部分
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            res_completions.append(text)


    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(gsm8k_ins, res_completions, gsm8k_answers)):
        doc = {'question': prompt}
        y_pred = extract_answer_number(completion)
        if y_pred != None:
            result.append(float(y_pred) == float(prompt_answer))
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)
    acc = sum(result) / (len(result) - len(invalid_outputs))
    print('len invalid outputs ====', len(invalid_outputs), ', valid_outputs===', invalid_outputs)
    print('start===', start, ', end====', end)
    print('gsm8k length====', len(result), ', gsm8k acc====', acc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)  # model path
    parser.add_argument("--data_file", type=str, default='')  # data path
    parser.add_argument("--start", type=int, default=0) #start index
    parser.add_argument("--end", type=int, default=MAX_INT)  # end index
    parser.add_argument("--batch_size", type=int, default=400)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=8)  # tensor_parallel_size
    return parser.parse_args()
if __name__ == "__main__":
    args = parse_args()
    gsm8k_test(model_name=args.model, data_path=args.data_file, start=args.start, end=args.end, batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size)
