#!/usr/bin/env python3
"""
生成蛋白-小分子对的JSON文件和蛋白-only的JSON文件
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


def parse_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    """
    解析FASTA文件，返回[(protein_name, sequence), ...]
    """
    proteins = []
    with open(fasta_path, 'r') as f:
        name = None
        sequence = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if name is not None:
                    proteins.append((name, ''.join(sequence)))
                name = line[1:].split()[0]  # 取第一个空格前的部分作为名字
                sequence = []
            else:
                sequence.append(line)
        if name is not None:
            proteins.append((name, ''.join(sequence)))
    return proteins


def parse_smi(smi_path: str) -> List[Tuple[str, str]]:
    """
    解析SMI文件，返回[(ligand_id, smiles), ...]
    支持格式：
    - SMILES ID
    - ID SMILES
    """
    ligands = []
    with open(smi_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                # 尝试判断哪个是SMILES（通常包含化学符号）
                if any(c in parts[0] for c in ['(', ')', '[', ']', '=', '#']):
                    smiles, ligand_id = parts[0], parts[1]
                else:
                    ligand_id, smiles = parts[0], parts[1]
                ligands.append((ligand_id, smiles))
            elif len(parts) == 1:
                # 只有SMILES，使用序号作为ID
                ligands.append((f"ligand_{len(ligands)+1}", parts[0]))
    return ligands


def generate_protein_ligand_json(
    proteins: List[Tuple[str, str]],
    ligands: List[Tuple[str, str]],
    msa_dir: str,
    output_path: str
) -> None:
    """
    生成蛋白-小分子对的JSON文件
    """
    entries = []
    for protein_name, sequence in proteins:
        for ligand_id, smiles in ligands:
            entry = {
                "name": f"{protein_name}/{ligand_id}",
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": sequence,
                            "count": 1,
                            "msa": {
                                "precomputed_msa_dir": msa_dir,
                                "pairing_db": "uniref100"
                            }
                        }
                    },
                    {
                        "ligand": {
                            "ligand": smiles,
                            "count": 1
                        }
                    }
                ]
            }
            entries.append(entry)
    
    with open(output_path, 'w') as f:
        json.dump(entries, f, indent=2)
    
    print(f"✓ 生成蛋白-小分子对JSON: {output_path}")
    print(f"  - 蛋白数量: {len(proteins)}")
    print(f"  - 小分子数量: {len(ligands)}")
    print(f"  - 总配对数: {len(entries)}")


def generate_protein_only_json(
    proteins: List[Tuple[str, str]],
    msa_dir: str,
    output_path: str
) -> None:
    """
    生成蛋白-only的JSON文件
    """
    entries = []
    for protein_name, sequence in proteins:
        entry = {
            "name": protein_name,
            "sequences": [
                {
                    "proteinChain": {
                        "sequence": sequence,
                        "count": 1,
                        "msa": {
                            "precomputed_msa_dir": msa_dir,
                            "pairing_db": "uniref100"
                        }
                    }
                }
            ]
        }
        entries.append(entry)
    
    with open(output_path, 'w') as f:
        json.dump(entries, f, indent=2)
    
    print(f"✓ 生成蛋白-only JSON: {output_path}")
    print(f"  - 蛋白数量: {len(proteins)}")


def main():
    parser = argparse.ArgumentParser(
        description="生成蛋白-小分子对的JSON文件和蛋白-only的JSON文件"
    )
    parser.add_argument(
        '--fasta', required=True,
        help='蛋白质FASTA文件路径'
    )
    parser.add_argument(
        '--smi', required=True,
        help='小分子SMI文件路径'
    )
    parser.add_argument(
        '--msa_dir', required=True,
        help='MSA文件目录路径（容器内路径）'
    )
    parser.add_argument(
        '--output_dir', required=True,
        help='输出目录路径'
    )
    parser.add_argument(
        '--project_name', required=True,
        help='项目名称（用于命名输出文件）'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 解析输入文件
    print("📖 解析输入文件...")
    proteins = parse_fasta(args.fasta)
    ligands = parse_smi(args.smi)
    
    # 生成JSON文件
    print("\n📝 生成JSON文件...")
    pair_json_path = output_dir / f"{args.project_name}.json"
    protein_json_path = output_dir / f"{args.project_name}_protein.json"
    
    generate_protein_ligand_json(proteins, ligands, args.msa_dir, str(pair_json_path))
    generate_protein_only_json(proteins, args.msa_dir, str(protein_json_path))
    
    print(f"\n✅ JSON文件生成完成！")


if __name__ == "__main__":
    main()
