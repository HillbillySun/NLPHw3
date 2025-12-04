import math
import os
from openai import OpenAI

# ========== 1. 准备 10 个 seed instruction–response 对 ==========
seed_data = [
    {
        "instruction": "Explain the difference between supervised and unsupervised learning.",
        "reference": "Supervised learning trains models on labeled data, while unsupervised learning uses unlabeled data to discover hidden structure or patterns.",
        "keywords": ["supervised learning", "unsupervised learning", "labels", "data"]
    },
    {
        "instruction": "Describe overfitting in one sentence.",
        "reference": "Overfitting happens when a model performs very well on the training set but generalizes poorly to new, unseen data.",
        "keywords": ["training set", "unseen data", "performance", "poor"]
    },
    {
        "instruction": "Briefly explain the main idea of gradient descent.",
        "reference": "Gradient descent iteratively updates model parameters in the opposite direction of the loss function's gradient to gradually reduce the loss.",
        "keywords": ["gradient", "loss function", "parameters", "update"]
    },
    {
        "instruction": "What is an activation function in a neural network? Give one example.",
        "reference": "An activation function introduces non-linearity into a neuron’s output, such as ReLU or Sigmoid.",
        "keywords": ["activation function", "non-linearity", "ReLU", "Sigmoid"]
    },
    {
        "instruction": "Why do we split a dataset into training and test sets?",
        "reference": "We split data into training and test sets to evaluate how well the model generalizes to unseen data instead of just memorizing the training data.",
        "keywords": ["training set", "test set", "generalization", "evaluate"]
    },
    {
        "instruction": "What is the learning rate and how does it affect training?",
        "reference": "The learning rate controls the step size of each parameter update; if it is too large training may diverge, and if it is too small convergence will be very slow.",
        "keywords": ["learning rate", "step size", "convergence", "diverge"]
    },
    {
        "instruction": "Explain TP, FP, TN, and FN in a confusion matrix.",
        "reference": "TP is true positive, FP is false positive, TN is true negative, and FN is false negative, representing the four combinations of predicted and actual labels.",
        "keywords": ["TP", "FP", "TN", "FN"]
    },
    {
        "instruction": "What is regularization and what problem does it address?",
        "reference": "Regularization adds a penalty term to the loss function to constrain model complexity and reduce overfitting.",
        "keywords": ["regularization", "penalty", "complexity", "overfitting"]
    },
    {
        "instruction": "Describe the main steps of k-means clustering.",
        "reference": "k-means repeatedly assigns samples to the nearest cluster center and then updates the cluster centers until convergence.",
        "keywords": ["k-means", "cluster centers", "samples", "convergence"]
    },
    {
        "instruction": "Explain the meaning of the ROC curve and AUC.",
        "reference": "The ROC curve shows the trade-off between true positive rate and false positive rate at different thresholds, and AUC is the area under this curve that summarizes the model’s overall discriminative ability.",
        "keywords": ["ROC", "AUC", "true positive rate", "false positive rate"]
    },
]


# ========== 2. 使用 Qwen API 真正生成候选回答 ==========

# 全局初始化 Qwen 客户端（阿里云百炼 OpenAI 兼容接口）
qwen_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 请确认你已经设置了这个环境变量
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def generate_candidates(instruction: str, num_candidates: int = 3):
    """
    通过 Qwen 模型生成若干候选回答。
    这里采用循环调用的方式，兼容性更好。
    如果你确认 chat.completions 支持 n 参数，也可以改成一次请求拿多条。
    """
    candidates = []
    for _ in range(num_candidates):
        resp = qwen_client.chat.completions.create(
            model="qwen-plus",  # 也可以换成 qwen-max / qwen3-8b 等
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful ML teaching assistant. 请用简洁中文回答。",
                },
                {"role": "user", "content": instruction},
            ],
            temperature=0.7,
        )
        candidates.append(resp.choices[0].message.content.strip())
    return candidates

# ========== 3. 定义启发式奖励函数 ==========

def tokenize(text: str):
    # 非严格分词：按空格和常见中英文标点简单切分
    for ch in "，。！？：；,.!?:":
        text = text.replace(ch, " ")
    tokens = [t for t in text.strip().split() if t]
    return tokens

def length_score(text: str, ideal_len: int = 30) -> float:
    """
    简单长度得分：长度越接近 ideal_len 越好。
    这里用一个平滑的函数，得分范围大致在 0 ~ 1。
    """
    n = len(tokenize(text))
    if n == 0:
        return 0.0
    return math.exp(-((n - ideal_len) ** 2) / (2 * (ideal_len / 2) ** 2))

def keyword_score(text: str, keywords) -> float:
    """
    关键词匹配率：命中多少关键词
    """
    if not keywords:
        return 0.0
    cnt = 0
    for kw in keywords:
        if kw in text:
            cnt += 1
    return cnt / len(keywords)

def jaccard_similarity(a: str, b: str) -> float:
    """
    Jaccard 相似度：|A∩B| / |A∪B|
    这里作为非常简化的“参考答案相似度”指标
    """
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta and not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union > 0 else 0.0

def reward_function(candidate: str, reference: str, keywords,
                    w_len: float = 0.3, w_kw: float = 0.3, w_sim: float = 0.4):
    """
    总 reward = 长度得分 * w_len + 关键词得分 * w_kw + Jaccard 相似度 * w_sim
    """
    ls = length_score(candidate)
    ks = keyword_score(candidate, keywords)
    sim = jaccard_similarity(candidate, reference)
    reward = w_len * ls + w_kw * ks + w_sim * sim
    return reward, (ls, ks, sim)

# ========== 4. 对每个指令生成候选、打分并筛选 top-1 ==========

def run_reward_based_filtering(num_candidates_per_prompt: int = 3):
    for idx, sample in enumerate(seed_data, start=1):
        instr = sample["instruction"]
        ref = sample["reference"]
        keywords = sample["keywords"]

        # 调用 Qwen 生成候选
        candidates = generate_candidates(instr, num_candidates=num_candidates_per_prompt)

        print("=" * 80)
        print(f"Prompt {idx}: {instr}")
        print(f"参考答案: {ref}")
        print("-" * 80)

        best_score = -1.0
        best_id = -1

        for i, cand in enumerate(candidates, start=1):
            score, (ls, ks, sim) = reward_function(cand, ref, keywords)
            print(f"Candidate {i}: {cand}")
            print(f"  -> Score = {score:.4f} "
                  f"(len={ls:.3f}, kw={ks:.3f}, sim={sim:.3f})")
            print()

            if score > best_score:
                best_score = score
                best_id = i

        print(f"Selected: Candidate {best_id}")
        print()

if __name__ == "__main__":
    # 可以在这里调整每个指令生成的候选数量
    run_reward_based_filtering(num_candidates_per_prompt=3)
