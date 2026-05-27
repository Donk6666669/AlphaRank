#!/usr/bin/env python3
"""
生成用于运行protenix predict的Shell脚本
"""

import argparse
from pathlib import Path


def generate_predict_script(
    json_path: str,
    output_dir: str,
    lmdb_path: str,
    gpu: int,
    seeds: int,
    script_path: str,
    start: int = 0,
    n_parallel: int = 8
) -> None:
    """
    生成用于生成LMDB的Shell脚本
    使用 runner/batch_inference.py predict --reduce
    运行AlphaFold/Protenix结构预测并保存embedding
    """
    script_content = f"""#!/bin/bash
export CUDA_VISIBLE_DEVICES={gpu}

python /project/pipeline/run_predict.py \\
    --input {json_path} \\
    --out_dir {output_dir} \\
    --lmdb {lmdb_path} \\
    --reduce \\
    --use_msa_server \\
    --seeds {seeds}
"""
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # 添加执行权限
    Path(script_path).chmod(0o755)
    
    print(f"✓ 生成Shell脚本: {script_path}")


def main():
    parser = argparse.ArgumentParser(
        description="生成用于运行protenix predict的Shell脚本"
    )
    parser.add_argument(
        '--project_name', required=True,
        help='项目名称'
    )
    parser.add_argument(
        '--json_dir', required=True,
        help='JSON文件所在目录'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='输出目录（容器内路径）'
    )
    parser.add_argument(
        '--lmdb_dir', required=True,
        help='LMDB输出目录（容器内路径）'
    )
    parser.add_argument(
        '--script_dir', required=True,
        help='Shell脚本输出目录'
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU编号（默认：0）'
    )
    parser.add_argument(
        '--seeds', type=int, default=101,
        help='随机种子（默认：101）'
    )
    parser.add_argument(
        '--start', type=int, default=0,
        help='起始索引（默认：0）'
    )
    parser.add_argument(
        '--n_parallel', type=int, default=8,
        help='并行数（默认：8）'
    )
    
    args = parser.parse_args()
    
    # 创建脚本输出目录
    script_dir = Path(args.script_dir)
    script_dir.mkdir(parents=True, exist_ok=True)
    
    json_dir = Path(args.json_dir)
    lmdb_dir = Path(args.lmdb_dir)
    
    # 生成蛋白-小分子对的脚本
    print("📝 生成Shell脚本...")
    pair_json = json_dir / f"{args.project_name}.json"
    pair_lmdb = lmdb_dir / f"{args.project_name}.lmdb"
    pair_script = script_dir / f"{args.project_name}_pair.sh"
    
    generate_predict_script(
        json_path=str(pair_json),
        output_dir=args.output_dir,
        lmdb_path=str(pair_lmdb),
        gpu=args.gpu,
        seeds=args.seeds,
        script_path=str(pair_script),
        start=args.start,
        n_parallel=args.n_parallel
    )
    
    # 生成蛋白-only的脚本
    protein_json = json_dir / f"{args.project_name}_protein.json"
    protein_lmdb = lmdb_dir / f"{args.project_name}_protein.lmdb"
    protein_script = script_dir / f"{args.project_name}_protein.sh"
    
    generate_predict_script(
        json_path=str(protein_json),
        output_dir=args.output_dir,
        lmdb_path=str(protein_lmdb),
        gpu=args.gpu,
        seeds=args.seeds,
        script_path=str(protein_script),
        start=args.start,
        n_parallel=args.n_parallel
    )
    
    print(f"\n✅ Shell脚本生成完成！")
    print(f"\n运行命令：")
    print(f"  bash {pair_script}")
    print(f"  bash {protein_script}")


if __name__ == "__main__":
    main()
