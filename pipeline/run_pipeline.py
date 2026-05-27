#!/usr/bin/env python3
"""
蛋白-小分子亲和力预测Pipeline主入口脚本

完整流程：
1. 生成JSON文件（蛋白-小分子对 + 蛋白-only）
2. 生成Shell脚本
3. 将JSON和Shell脚本复制到单一运行容器可访问的/data目录
4. 在同一个容器中运行protenix predict生成pair embedding LMDB
5. 在同一个容器中运行transform.py给LMDB添加名字，并从pair embedding抽取protein LMDB
6. 在同一个容器中运行inference得到最终CSV

使用示例：
    python pipeline/run_pipeline.py \\
        --fasta /path/to/protein.fasta \\
        --smi /path/to/ligands.smi \\
        --msa_dir /data/msa_dir \\
        --project_name my_project \\
        --output_dir /data/output \\
        --lmdb_dir /data/lmdb \\
        --gpu 0 \\
        --mode full
"""

import argparse
import subprocess
import shutil
import sys
from pathlib import Path


class PipelineRunner:
    def __init__(
        self,
        fasta_path: str,
        smi_path: str,
        msa_dir: str,
        project_name: str,
        output_dir_container: str,
        lmdb_dir_container: str,
        gpu: int,
        seeds: int,
        mode: str,
        ckpt_path: str = "/log/train/alpharank_hyper_decoy_new/DataModule.HYPMLPPairRanker.ListListRankCriterion.2026-01-07_10-59-36/checkpoints/ema_epoch40-49.ckpt",
        container_name: str = "alpharank_hyper_decoy_wukelin",
        host_project_dir: str = "/home/wukelin/ProtenixAffinity2",
        host_data_dir: str = "/msa/data",
        source_host_data_dir: str = "/data/protein"
    ):
        self.fasta_path = fasta_path
        self.smi_path = smi_path
        self.msa_dir = msa_dir
        self.project_name = project_name
        self.output_dir_container = output_dir_container
        self.lmdb_dir_container = lmdb_dir_container
        self.gpu = gpu
        self.seeds = seeds
        self.mode = mode
        self.ckpt_path = ckpt_path
        self.container_name = container_name
        self.host_project_dir = host_project_dir
        self.host_data_dir = host_data_dir
        self.source_host_data_dir = source_host_data_dir
        
        # 设置工作目录
        self.pipeline_dir = Path(__file__).parent
        self.workspace_dir = self.pipeline_dir.parent
        self.json_output_dir = self.workspace_dir / "output" / project_name / "json"
        self.script_output_dir = self.workspace_dir / "output" / project_name / "scripts"
        
    def run_command(self, cmd: str, show_output: bool = True) -> int:
        """运行命令并返回退出码"""
        print(f"\n🔧 执行命令: {cmd}")
        if show_output:
            result = subprocess.run(cmd, shell=True)
        else:
            result = subprocess.run(
                cmd, shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        return result.returncode
    
    def run_in_container(self, cmd: str) -> int:
        """在单一运行容器中运行命令"""
        docker_cmd = f"docker exec -i {self.container_name} /bin/zsh -c 'cd /project && {cmd}'"
        return self.run_command(docker_cmd)
    
    def container_data_path_to_host(self, container_path: str) -> str:
        """将容器内/data路径转换为当前运行容器对应的宿主机路径"""
        if container_path == '/data':
            return self.host_data_dir
        if container_path.startswith('/data/'):
            return container_path.replace('/data/', f'{self.host_data_dir}/', 1)
        return container_path
    
    def ensure_msa_available_in_container(self) -> bool:
        """确保MSA目录在单一运行容器的/data挂载下可见"""
        if not self.msa_dir.startswith('/data'):
            return True

        target_host = Path(self.container_data_path_to_host(self.msa_dir))
        if target_host.exists():
            print(f"✓ MSA目录已在运行容器挂载中: {target_host}")
            return True

        rel_path = self.msa_dir[len('/data/'):].strip('/') if self.msa_dir != '/data' else ''
        source_host = Path(self.source_host_data_dir) / rel_path
        if source_host.exists():
            print(f"🔄 MSA目录在运行容器挂载中不存在，正在从旧挂载复制:")
            print(f"   {source_host} -> {target_host}")
            target_host.parent.mkdir(parents=True, exist_ok=True)
            if source_host.is_dir():
                shutil.copytree(source_host, target_host)
            else:
                shutil.copy2(source_host, target_host)
            return True

        print(f"❌ MSA目录不存在: {target_host}")
        print(f"   也未在旧挂载中找到: {source_host}")
        return False
    
    def sync_helper_scripts_to_container_project(self) -> bool:
        """把单容器运行所需的辅助脚本同步到容器/project映射目录"""
        print("\n🔄 同步单容器运行所需脚本到容器 /project ...")
        helper_files = [
            (self.pipeline_dir / "run_predict.py", Path(self.host_project_dir) / "pipeline" / "run_predict.py"),
            (self.workspace_dir / "test_scripts" / "transform.py", Path(self.host_project_dir) / "test_scripts" / "transform.py"),
            (self.workspace_dir / "test_scripts" / "openbind.py", Path(self.host_project_dir) / "test_scripts" / "openbind.py"),
        ]
        for src, dest in helper_files:
            if not src.exists():
                print(f"❌ 辅助脚本不存在: {src}")
                return False
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            print(f"✓ {src.name} -> {dest}")
        return True
    
    def step1_generate_json(self) -> bool:
        """步骤1: 生成JSON文件"""
        print("\n" + "="*60)
        print("步骤1: 生成JSON文件")
        print("="*60)
        
        cmd = (
            f"python {self.pipeline_dir}/generate_json.py "
            f"--fasta {self.fasta_path} "
            f"--smi {self.smi_path} "
            f"--msa_dir {self.msa_dir} "
            f"--output_dir {self.json_output_dir} "
            f"--project_name {self.project_name}"
        )
        
        ret = self.run_command(cmd)
        if ret != 0:
            print("❌ 生成JSON文件失败")
            return False
        return True
    
    def step2_generate_scripts(self) -> bool:
        """步骤2: 生成Shell脚本"""
        print("\n" + "="*60)
        print("步骤2: 生成Shell脚本")
        print("="*60)
        
        cmd = (
            f"python {self.pipeline_dir}/generate_sh.py "
            f"--project_name {self.project_name} "
            f"--json_dir {self.lmdb_dir_container} "
            f"--output_dir {self.output_dir_container} "
            f"--lmdb_dir {self.lmdb_dir_container} "
            f"--script_dir {self.script_output_dir} "
            f"--gpu {self.gpu} "
            f"--seeds {self.seeds}"
        )
        
        ret = self.run_command(cmd)
        if ret != 0:
            print("❌ 生成Shell脚本失败")
            return False
        return True
    
    def step3_prepare_container_inputs(self) -> bool:
        """步骤3: 复制JSON和Shell脚本到单一运行容器可访问目录"""
        print("\n" + "="*60)
        print("步骤3: 准备单容器输入文件")
        print("="*60)
        
        if not self.sync_helper_scripts_to_container_project():
            return False

        if not self.ensure_msa_available_in_container():
            return False

        # 计算宿主机路径（单一运行容器: /data -> 宿主机: /msa/data）
        output_dir_host = self.container_data_path_to_host(self.output_dir_container)
        Path(output_dir_host).mkdir(parents=True, exist_ok=True)
        lmdb_dir_host = self.container_data_path_to_host(self.lmdb_dir_container)
        Path(lmdb_dir_host).mkdir(parents=True, exist_ok=True)
        
        for json_file in self.json_output_dir.glob("*.json"):
            dest = Path(lmdb_dir_host) / json_file.name
            shutil.copy(json_file, dest)
            print(f"✓ 复制 {json_file.name} -> {dest}")
        
        for script_file in self.script_output_dir.glob("*.sh"):
            dest = Path(lmdb_dir_host) / script_file.name
            shutil.copy(script_file, dest)
            dest.chmod(0o755)
            print(f"✓ 复制 {script_file.name} -> {dest}")
        
        return True
    
    def step4_run_protenix_predict(self) -> bool:
        """步骤4: 运行protenix predict生成LMDB"""
        print("\n" + "="*60)
        print("步骤4: 在单容器中运行protenix predict生成LMDB")
        print("="*60)
        
        # 运行蛋白-小分子对的预测
        pair_script_container = f"{self.lmdb_dir_container}/{self.project_name}_pair.sh"
        print(f"\n🚀 运行蛋白-小分子对LMDB生成...")
        ret = self.run_in_container(f"bash {pair_script_container}")
        if ret != 0:
            print("❌ 蛋白-小分子对LMDB生成失败")
            return False
        
        print("✅ LMDB生成完成")
        return True
    
    def step5_run_transform(self) -> bool:
        """步骤5: 在单容器中运行transform.py给LMDB添加名字"""
        print("\n" + "="*60)
        print("步骤5: 运行transform.py")
        print("="*60)
        
        pair_lmdb = f"{self.lmdb_dir_container}/{self.project_name}.lmdb"
        pair_json = f"{self.lmdb_dir_container}/{self.project_name}.json"
        new_lmdb  = f"{self.lmdb_dir_container}/{self.project_name}_named.lmdb"
        protein_lmdb = f"{self.lmdb_dir_container}/{self.project_name}_protein.lmdb"
        
        cmd = (
            f"python test_scripts/transform.py"
            f" --old_lmdb {pair_lmdb}"
            f" --json_path {pair_json}"
            f" --new_lmdb {new_lmdb}"
            f" --protein_lmdb {protein_lmdb}"
        )
        ret = self.run_in_container(cmd)
        if ret != 0:
            print("❌ transform.py 运行失败")
            return False
        print("✅ LMDB名字转换完成")
        return True
    
    def step6_run_inference(self) -> bool:
        """步骤6: 在单容器中运行inference"""
        print("\n" + "="*60)
        print("步骤6: 运行inference")
        print("="*60)
        
        pair_lmdb  = f"{self.lmdb_dir_container}/{self.project_name}_named.lmdb/"
        prot_lmdb  = f"{self.lmdb_dir_container}/{self.project_name}_protein.lmdb/"
        # 输出到单容器内的 /project/output/<project_name>/results/
        output_dir = f"/project/output/{self.project_name}/results"
        
        cmd = (
            f"CUDA_VISIBLE_DEVICES={self.gpu} PYTHONPATH=. python test_scripts/openbind.py"
            f" --data_paths {pair_lmdb}"
            f" --protein_path {prot_lmdb}"
            f" --ckpt_path {self.ckpt_path}"
            f" --output_dir {output_dir}"
        )
        ret = self.run_in_container(cmd)
        if ret != 0:
            print("❌ Inference 失败")
            return False
        
        # 把结果从容器/project映射目录复制回当前codebase的output目录
        result_host_src = Path(self.host_project_dir) / "output" / self.project_name / "results"
        result_host_dst = Path(self.workspace_dir) / "output" / self.project_name / "results"
        if result_host_src.exists():
            if result_host_dst.exists():
                shutil.rmtree(result_host_dst)
            shutil.copytree(result_host_src, result_host_dst)
            print(f"✅ Inference完成！结果已保存至: {result_host_dst}")
            for csv in result_host_dst.glob("*.csv"):
                print(f"   📄 {csv}")
        else:
            print(f"✅ Inference完成！结果在容器路径: {output_dir}")
        
        return True
    
    def run(self) -> bool:
        """运行完整pipeline"""
        print("\n" + "="*60)
        print(f"🚀 开始运行Pipeline: {self.project_name}")
        print(f"   模式: {self.mode}")
        print("="*60)
        
        if self.mode in ['full', 'generate_only']:
            if not self.step1_generate_json():
                return False
            if not self.step2_generate_scripts():
                return False
        
        if self.mode == 'generate_only':
            print("\n✅ JSON和脚本生成完成（仅生成模式）")
            return True
        
        if self.mode == 'full':
            if not self.step3_prepare_container_inputs():
                return False
            if not self.step4_run_protenix_predict():
                return False
            if not self.step5_run_transform():
                return False
            if not self.step6_run_inference():
                return False
        
        if self.mode == 'from_lmdb':
            if not self.step3_prepare_container_inputs():
                return False
            named_lmdb_host = Path(self.container_data_path_to_host(self.lmdb_dir_container)) / f"{self.project_name}_named.lmdb"
            if not named_lmdb_host.exists():
                if not self.step5_run_transform():
                    return False
            if not self.step6_run_inference():
                return False
        
        print("\n" + "="*60)
        print("✅ Pipeline运行完成！")
        print("="*60)
        return True


def main():
    parser = argparse.ArgumentParser(
        description="蛋白-小分子亲和力预测Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整运行
  python pipeline/run_pipeline.py \\
      --fasta /path/to/protein.fasta \\
      --smi /path/to/ligands.smi \\
      --msa_dir /data/msa_dir \\
      --project_name my_project \\
    --output_dir /data/output \\
    --lmdb_dir /data/lmdb \\
      --gpu 0 \\
      --mode full

  # 仅生成JSON和脚本
  python pipeline/run_pipeline.py ... --mode generate_only

  # 从已有LMDB开始
  python pipeline/run_pipeline.py ... --mode from_lmdb
        """
    )
    
    parser.add_argument('--fasta', required=True, help='蛋白质FASTA文件路径')
    parser.add_argument('--smi', required=True, help='小分子SMI文件路径')
    parser.add_argument('--msa_dir', required=True, help='MSA目录（单一运行容器内路径）')
    parser.add_argument('--project_name', required=True, help='项目名称')
    parser.add_argument('--output_dir', '--output_dir_c1', dest='output_dir', required=True, help='单一运行容器输出目录（兼容旧参数名 --output_dir_c1）')
    parser.add_argument('--lmdb_dir', '--lmdb_dir_c1', dest='lmdb_dir', required=True, help='单一运行容器LMDB目录（兼容旧参数名 --lmdb_dir_c1）')
    parser.add_argument('--lmdb_dir_c2', help='旧版双容器参数，单容器模式下忽略')
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号（默认：0）')
    parser.add_argument('--seeds', type=int, default=101, help='随机种子（默认：101）')
    parser.add_argument(
        '--mode',
        choices=['full', 'generate_only', 'from_lmdb'],
        default='full',
        help='运行模式（默认：full）'
    )
    parser.add_argument('--ckpt_path',
        default='/log/train/alpharank_hyper_decoy_new/DataModule.HYPMLPPairRanker.ListListRankCriterion.2026-01-07_10-59-36/checkpoints/ema_epoch40-49.ckpt',
        help='模型checkpoint路径（单一运行容器内路径）')
    parser.add_argument('--container', '--container2', dest='container', default='alpharank_hyper_decoy_wukelin', help='单一运行容器名称（兼容旧参数名 --container2）')
    parser.add_argument('--container1', help='旧版双容器参数，单容器模式下忽略')
    parser.add_argument('--host_project_dir', default='/home/wukelin/ProtenixAffinity2', help='单一运行容器/project对应的宿主机目录')
    parser.add_argument('--host_data_dir', default='/msa/data', help='单一运行容器/data对应的宿主机目录')
    parser.add_argument('--source_host_data_dir', default='/data/protein', help='旧/data挂载对应的宿主机目录；当MSA尚未迁移到单容器/data时用于自动复制')
    
    args = parser.parse_args()
    
    runner = PipelineRunner(
        fasta_path=args.fasta,
        smi_path=args.smi,
        msa_dir=args.msa_dir,
        project_name=args.project_name,
        output_dir_container=args.output_dir,
        lmdb_dir_container=args.lmdb_dir,
        ckpt_path=args.ckpt_path,
        gpu=args.gpu,
        seeds=args.seeds,
        mode=args.mode,
        container_name=args.container,
        host_project_dir=args.host_project_dir,
        host_data_dir=args.host_data_dir,
        source_host_data_dir=args.source_host_data_dir
    )
    
    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
