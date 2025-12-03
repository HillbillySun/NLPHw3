import json
import argparse

def cut_json_array(in_file, out_file, k):
    """输入是一个大数组的 .json 文件"""
    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)          # data 是一个 list
    cut_data = data[:k]
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(cut_data, f, ensure_ascii=False, indent=2)


def cut_jsonl(in_file, out_file, k):
    """输入是 jsonl，一行一条 json"""
    lines = []
    with open(in_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= k:
                break
            lines.append(line)

    with open(out_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入文件路径（.json 或 .jsonl）")
    parser.add_argument("--output", required=True, help="输出文件路径")
    parser.add_argument("--k", type=int, required=True, help="保留前 k 条数据")
    parser.add_argument("--format", choices=["json", "jsonl"], required=True,
                        help="输入文件格式：json（大数组）或 jsonl（一行一条）")
    args = parser.parse_args()

    if args.format == "json":
        cut_json_array(args.input, args.output, args.k)
    else:
        cut_jsonl(args.input, args.output, args.k)


if __name__ == "__main__":
    main()
