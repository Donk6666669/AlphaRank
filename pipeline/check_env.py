#!/usr/bin/env python3
"""
验证输入文件格式和环境配置
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_file_exists(filepath: str, description: str) -> bool:
    """检查文件是否存在"""
    path = Path(filepath)
    if path.exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} 不存在: {filepath}")
        return False


def validate_fasta(fasta_path: str) -> bool:
    """验证FASTA文件格式"""
    print(f"\n📖 验证FASTA文件: {fasta_path}")
    
    if not check_file_exists(fasta_path, "FASTA文件"):
        return False
    
    try:
        with open(fasta_path, 'r') as f:
            lines = f.readlines()
        
        protein_count = 0
        in_sequence = False
        current_name = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                protein_count += 1
                current_name = line[1:].split()[0]
                in_sequence = True
                print(f"   找到蛋白: {current_name}")
            elif line and in_sequence:
                # 检查是否包含非法字符
                valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
                invalid_chars = set(line.upper()) - valid_aa
                if invalid_chars:
                    print(f"   ⚠️  警告: 序列包含非标准氨基酸: {invalid_chars}")
        
        if protein_count == 0:
            print("   ❌ 错误: 未找到任何蛋白序列")
            return False
        
        print(f"   ✅ 共找到 {protein_count} 个蛋白序列")
        return True
    
    except Exception as e:
        print(f"   ❌ 读取FASTA文件失败: {e}")
        return False


def validate_smi(smi_path: str) -> bool:
    """验证SMI文件格式"""
    print(f"\n📖 验证SMI文件: {smi_path}")
    
    if not check_file_exists(smi_path, "SMI文件"):
        return False
    
    try:
        with open(smi_path, 'r') as f:
            lines = f.readlines()
        
        ligand_count = 0
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 2:
                print(f"   ⚠️  警告: 第{i}行格式可能不正确（需要ID和SMILES）")
                continue
            
            ligand_count += 1
            if ligand_count <= 3:  # 只显示前3个
                print(f"   找到小分子: {parts[0]} | {parts[1][:30]}...")
        
        if ligand_count == 0:
            print("   ❌ 错误: 未找到任何小分子")
            return False
        
        print(f"   ✅ 共找到 {ligand_count} 个小分子")
        return True
    
    except Exception as e:
        print(f"   ❌ 读取SMI文件失败: {e}")
        return False


def check_docker_containers() -> bool:
    """检查Docker容器状态"""
    print("\n🐋 检查Docker容器...")
    
    containers = ["alpharank_hyper_decoy_wukelin"]
    all_ok = True
    
    for container in containers:
        try:
            result = subprocess.run(
                f"docker ps --filter name={container} --format '{{{{.Names}}}}'",
                shell=True,
                capture_output=True,
                text=True
            )
            if container in result.stdout:
                print(f"   ✅ 容器运行中: {container}")
            else:
                print(f"   ❌ 容器未运行: {container}")
                all_ok = False
        except Exception as e:
            print(f"   ❌ 检查容器失败: {e}")
            all_ok = False
    
    return all_ok


def check_gpu() -> bool:
    """检查GPU状态"""
    print("\n🎮 检查GPU...")
    
    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                print(f"   ✅ GPU: {line}")
            return True
        else:
            print("   ❌ 无法访问GPU")
            return False
    except Exception as e:
        print(f"   ❌ 检查GPU失败: {e}")
        return False


def check_disk_space(paths: list) -> bool:
    """检查磁盘空间"""
    print("\n💾 检查磁盘空间...")
    
    all_ok = True
    for path in paths:
        try:
            result = subprocess.run(
                f"df -h {path}",
                shell=True,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    print(f"   {path}")
                    print(f"   {lines[1]}")
            else:
                print(f"   ⚠️  无法检查路径: {path}")
        except Exception as e:
            print(f"   ❌ 检查磁盘空间失败: {e}")
            all_ok = False
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="验证ProtenixAffinity Pipeline的输入和环境"
    )
    parser.add_argument('--fasta', help='FASTA文件路径')
    parser.add_argument('--smi', help='SMI文件路径')
    parser.add_argument('--check-docker', action='store_true', help='检查Docker容器')
    parser.add_argument('--check-gpu', action='store_true', help='检查GPU状态')
    parser.add_argument('--check-disk', nargs='+', help='检查磁盘空间的路径')
    parser.add_argument('--full', action='store_true', help='执行所有检查')
    
    args = parser.parse_args()
    
    if not any([args.fasta, args.smi, args.check_docker, args.check_gpu, args.check_disk, args.full]):
        parser.print_help()
        sys.exit(1)
    
    print("="*60)
    print("ProtenixAffinity Pipeline - 环境验证")
    print("="*60)
    
    all_ok = True
    
    # 验证输入文件
    if args.fasta or args.full:
        if args.fasta:
            all_ok &= validate_fasta(args.fasta)
        else:
            print("\n⚠️  未指定FASTA文件，跳过验证")
    
    if args.smi or args.full:
        if args.smi:
            all_ok &= validate_smi(args.smi)
        else:
            print("\n⚠️  未指定SMI文件，跳过验证")
    
    # 检查环境
    if args.check_docker or args.full:
        all_ok &= check_docker_containers()
    
    if args.check_gpu or args.full:
        all_ok &= check_gpu()
    
    if args.check_disk or args.full:
        disk_paths = args.check_disk if args.check_disk else ['/msa/data']
        all_ok &= check_disk_space(disk_paths)
    
    # 总结
    print("\n" + "="*60)
    if all_ok:
        print("✅ 所有检查通过！可以开始运行Pipeline")
    else:
        print("❌ 部分检查未通过，请修复后再运行")
    print("="*60)
    
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
