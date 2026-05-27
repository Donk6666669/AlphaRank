#!/usr/bin/env python3
"""
Protenix predict 包装脚本
调用 runner/batch_inference.py 中的 inference_jsons 函数
生成蛋白-小分子 embedding LMDB

使用方式：
    python pipeline/run_predict.py \\
        --input /data/.../TestRun.json \\
        --out_dir /data/.../output \\
        --lmdb /data/.../TestRun.lmdb \\
        --reduce \\
        --use_msa_server \\
        --seeds 101
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="运行 protenix predict 生成 embedding LMDB")
    parser.add_argument("--input", required=True, help="JSON 输入文件路径")
    parser.add_argument("--out_dir", default="./output", help="结构预测输出目录")
    parser.add_argument("--lmdb", default=None, help="embedding LMDB 输出路径")
    parser.add_argument("--reduce", action="store_true", help="保存 reduced embedding (5-tuple)")
    parser.add_argument("--use_msa_server", action="store_true", help="使用 MSA 服务器（有 precomputed_msa_dir 时不触发实际请求）")
    parser.add_argument("--use_esm", action="store_true", help="使用 ESM embedding")
    parser.add_argument("--seeds", default="101", help="推断随机种子，逗号分隔")
    parser.add_argument("--start", type=int, default=0, help="起始索引")
    parser.add_argument("--end", type=int, default=None, help="结束索引")
    args = parser.parse_args()

    seeds = list(map(int, args.seeds.split(",")))

    from runner.batch_inference import inference_jsons, init_logging
    init_logging()
    inference_jsons(
        json_file=args.input,
        out_dir=args.out_dir,
        use_msa_server=args.use_msa_server,
        seeds=seeds,
        use_esm=args.use_esm,
        lmdb=args.lmdb,
        start=args.start,
        end=args.end,
        reduce=args.reduce,
    )


if __name__ == "__main__":
    main()
