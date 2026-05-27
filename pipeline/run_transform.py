#!/usr/bin/env python3
"""
自动运行transform.py的包装脚本
"""

import argparse
import sys
from pathlib import Path


def run_transform(
    old_lmdb: str,
    json_path: str,
    new_lmdb: str,
    map_size: int = 20_000_000_000
):
    """
    动态生成并执行transform.py
    """
    # 读取原始transform.py
    transform_template = Path(__file__).parent.parent / "test_scripts" / "transform.py"
    with open(transform_template, 'r') as f:
        content = f.read()
    
    # 替换路径
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if line.startswith('OLD_LMDB ='):
            new_lines.append(f'OLD_LMDB = "{old_lmdb}"')
        elif line.startswith('JSON_PATH ='):
            new_lines.append(f'JSON_PATH = "{json_path}"')
        elif line.startswith('NEW_LMDB ='):
            new_lines.append(f'NEW_LMDB = "{new_lmdb}"')
        elif line.startswith('MAP_SIZE ='):
            new_lines.append(f'MAP_SIZE = {map_size}')
        else:
            new_lines.append(line)
    
    # 执行修改后的代码
    modified_content = '\n'.join(new_lines)
    exec(modified_content, {'__name__': '__main__'})


def main():
    parser = argparse.ArgumentParser(
        description="运行transform.py给LMDB添加名字"
    )
    parser.add_argument('--old_lmdb', required=True, help='原始LMDB路径')
    parser.add_argument('--json_path', required=True, help='JSON文件路径')
    parser.add_argument('--new_lmdb', required=True, help='新LMDB输出路径')
    parser.add_argument('--map_size', type=int, default=20_000_000_000, help='LMDB大小')
    
    args = parser.parse_args()
    
    print(f"🔧 运行transform.py")
    print(f"   OLD_LMDB: {args.old_lmdb}")
    print(f"   JSON_PATH: {args.json_path}")
    print(f"   NEW_LMDB: {args.new_lmdb}")
    
    try:
        run_transform(args.old_lmdb, args.json_path, args.new_lmdb, args.map_size)
        print("✅ Transform完成")
    except Exception as e:
        print(f"❌ Transform失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
