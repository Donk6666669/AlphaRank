#  python scripts/colabfold_msa.py \
#     /data_hdd/protein/rerank/protenix/chembl_bdb/chembl_bdb_unique_sequences.fasta \
#     /data_ssd/protein/AIRFold/datasets_2024/database \
#     /data_hdd/protein/rerank/protenix/chembl_bdb/msa/data \
#     --db1 uniref30_2302_db \
#     --db3 colabfold_envdb_202108_db \
#     --mmseqs /data_hdd/home/casp15/code/mmseqs_test//MMseqs2/build/bin/mmseqs \
#     --db_load_mode 2


# # 13
# python scripts/preprocess/json2feature.py \
#     --json_path /data/rerank/protenix/chembl_bdb/chembl_bdb_protenix.json \
#     --lmdb_path /data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb \
#     --n_parallel 32 \
#     --start 0 \
#     --end 515638
# python scripts/preprocess/json2feature.py \
#     --json_path /data/rerank/protenix/chembl_bdb/chembl_bdb_protenix.json \
#     --lmdb_path /data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb \
#     --n_parallel 32 \
#     --start 0 \
#     --end 400000

# # 19
# python scripts/preprocess/json2feature.py \
#     --json_path /data/rerank/protenix/chembl_bdb/chembl_bdb_protenix.json \
#     --lmdb_path /data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb \
#     --n_parallel 32 \
#     --start 515638 \
#     --end 915638

# # 12
# python scripts/preprocess/json2feature.py \
#     --json_path /data/rerank/protenix/chembl_bdb/chembl_bdb_protenix.json \
#     --lmdb_path /data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb \
#     --n_parallel 32 \
#     --start 915638 \
#     --end 1315638
# python scripts/preprocess/json2feature.py \
#     --json_path /data/rerank/protenix/chembl_bdb/chembl_bdb_protenix.json \
#     --lmdb_path /data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb \
#     --n_parallel 32 \
#     --start 915638 \
#     --end 1215638

# # 46
# python scripts/preprocess/json2feature.py \
#     --json_path /data/rerank/protenix/chembl_bdb/chembl_bdb_protenix.json \
#     --lmdb_path /data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb \
#     --n_parallel 32 \
#     --start 1315638 \
#     --end 1715638

# # 47
# python scripts/preprocess/json2feature.py \
#     --json_path /data/rerank/protenix/chembl_bdb/chembl_bdb_protenix.json \
#     --lmdb_path /data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb \
#     --n_parallel 32 \
#     --start 1715638 \
#     --end 2115638

# python scripts/preprocess/json2feature.py \
#     --json_path /data/rerank/protenix/chembl_bdb/chembl_bdb_protenix.json \
#     --lmdb_path /data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb \
#     --n_parallel 32 \
#     --start 400000 \
#     --end 515638

# # 48
# python scripts/preprocess/json2feature.py \
#     --json_path /data/rerank/protenix/chembl_bdb/chembl_bdb_protenix.json \
#     --lmdb_path /data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb \
#     --n_parallel 32 \
#     --start 2115638 \
#     --end 2515638
# python scripts/preprocess/json2feature.py \
#     --json_path /data/rerank/protenix/chembl_bdb/chembl_bdb_protenix.json \
#     --lmdb_path /data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb \
#     --n_parallel 32 \
#     --start 1215638 \
#     --end 1315638


# # pair

# CUDA_VISIBLE_DEVICES=3 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/outpput \
#     --seeds 101 \
#     --use_msa_server \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/pair/sample_3w_pair_1.lmdb \
#     --start 0 \
#     --end 5000

# CUDA_VISIBLE_DEVICES=2 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/pair/sample_3w_pair_2.lmdb \
#     --start 5000 \
#     --end 10000

# CUDA_VISIBLE_DEVICES=1 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/pair/sample_3w_pair_3.lmdb \
#     --start 10000 \
#     --end 15000

# CUDA_VISIBLE_DEVICES=0 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/pair/sample_3w_pair_4.lmdb \
#     --start 15000 \
#     --end 20000

# CUDA_VISIBLE_DEVICES=0 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/pair/sample_3w_pair_5.lmdb \
#     --start 20000 \
#     --end 25000

# CUDA_VISIBLE_DEVICES=1 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/pair/sample_3w_pair_6.lmdb \
#     --start 25000 \
#     --end 30000

# CUDA_VISIBLE_DEVICES=7 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/pair/sample_3w_pair_7.lmdb \
#     --start 30000

# # triplet

# CUDA_VISIBLE_DEVICES=2 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/triplet/sample_3w_triplet_1.lmdb \
#     --start 0 \
#     --end 5000

# CUDA_VISIBLE_DEVICES=3 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/triplet/sample_3w_triplet_2.lmdb \
#     --start 5000 \
#     --end 10000

# CUDA_VISIBLE_DEVICES=4 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/triplet/sample_3w_triplet_3.lmdb \
#     --start 10000 \
#     --end 15000

# CUDA_VISIBLE_DEVICES=5 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/triplet/sample_3w_triplet_4.lmdb \
#     --start 15000 \
#     --end 20000

# CUDA_VISIBLE_DEVICES=6 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/triplet/sample_3w_triplet_5.lmdb \
#     --start 20000 \
#     --end 25000

# CUDA_VISIBLE_DEVICES=7 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/triplet/sample_3w_triplet_6.lmdb \
#     --start 25000

# # reduced pair
# CUDA_VISIBLE_DEVICES=0 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/pair/sample_30w_pair_1.lmdb \
#     --start 0 \
#     --end 25000

# CUDA_VISIBLE_DEVICES=1 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/pair/sample_30w_pair_2.lmdb \
#     --start 25000 \
#     --end 50000

# CUDA_VISIBLE_DEVICES=2 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/pair/sample_30w_pair_3.lmdb \
#     --start 50000 \
#     --end 75000

# CUDA_VISIBLE_DEVICES=3 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/pair/sample_30w_pair_4.lmdb \
#     --start 75000 \
#     --end 100000

# CUDA_VISIBLE_DEVICES=0 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/pair/sample_30w_pair_5.lmdb \
#     --start 100000 \
#     --end 125000

# CUDA_VISIBLE_DEVICES=1 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/pair/sample_30w_pair_6.lmdb \
#     --start 125000 \
#     --end 150000

# CUDA_VISIBLE_DEVICES=2 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/pair/sample_30w_pair_7.lmdb \
#     --start 150000 \
#     --end 175000

# CUDA_VISIBLE_DEVICES=3 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/pair/sample_30w_pair_8.lmdb \
#     --start 175000 \
#     --end 200000

# CUDA_VISIBLE_DEVICES=4 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/pair/sample_30w_pair_9.lmdb \
#     --start 200000 \
#     --end 225000

# CUDA_VISIBLE_DEVICES=5 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/pair/sample_30w_pair_10.lmdb \
#     --start 225000

# # reduced triplet
# CUDA_VISIBLE_DEVICES=6 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/triplet/sample_30w_triplet_1.lmdb \
#     --start 0 \
#     --end 25000

# CUDA_VISIBLE_DEVICES=7 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/triplet/sample_30w_triplet_2.lmdb \
#     --start 25000 \
#     --end 50000

# CUDA_VISIBLE_DEVICES=4 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/triplet/sample_30w_triplet_3.lmdb \
#     --start 50000 \
#     --end 75000

# CUDA_VISIBLE_DEVICES=5 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/triplet/sample_30w_triplet_4.lmdb \
#     --start 75000 \
#     --end 100000

# CUDA_VISIBLE_DEVICES=6 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/triplet/sample_30w_triplet_5.lmdb \
#     --start 100000 \
#     --end 125000

# CUDA_VISIBLE_DEVICES=7 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/triplet/sample_30w_triplet_6.lmdb \
#     --start 125000 \
#     --end 150000

# CUDA_VISIBLE_DEVICES=4 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/triplet/sample_30w_triplet_7.lmdb \
#     --start 150000 \
#     --end 175000

# CUDA_VISIBLE_DEVICES=5 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/triplet/sample_30w_triplet_8.lmdb \
#     --start 175000 \
#     --end 200000

# CUDA_VISIBLE_DEVICES=6 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/triplet/sample_30w_triplet_9.lmdb \
#     --start 200000 \
#     --end 225000

# CUDA_VISIBLE_DEVICES=7 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/triplet/sample_30w_triplet_10.lmdb \
#     --start 225000 \
#     --end 250000

# CUDA_VISIBLE_DEVICES=0 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/triplet/sample_30w_triplet_11.lmdb \
#     --start 250000 \
#     --end 275000

# CUDA_VISIBLE_DEVICES=1 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_30w/sample_30w_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_30w/output \
#     --seeds 101 \
#     --use_msa_server \
#     --reduce \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_30w/triplet/sample_30w_triplet_12.lmdb \
#     --start 275000 \
#     --end 300000

# # sample val pair
# CUDA_VISIBLE_DEVICES=6 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_val/sample_val_pair.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_val/output \
#     --seeds 101 \
#     --use_msa_server \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_val/sample_val_pair.lmdb

# # sample val triplet
# CUDA_VISIBLE_DEVICES=7 protenix predict --input /data/rerank/protenix/chembl_bdb/sample_val/sample_val_triplet.json \
#     --out_dir /data/rerank/protenix/chembl_bdb/sample_val/output \
#     --seeds 101 \
#     --use_msa_server \
#     --lmdb /data/rerank/protenix/chembl_bdb/sample_val/sample_val_triplet.lmdb

# pair with 5 recycles

protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_pair.json \
    --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
    --seeds 101 \
    --use_msa_server \
    --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/pair_5recycle/sample_3w_pair_1.lmdb \
    --start 0 \
    --end 5000

protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_pair.json \
    --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
    --seeds 101 \
    --use_msa_server \
    --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/pair_5recycle/sample_3w_pair_2.lmdb \
    --start 5000 \
    --end 10000

protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_pair.json \
    --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
    --seeds 101 \
    --use_msa_server \
    --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/pair_5recycle/sample_3w_pair_3.lmdb \
    --start 10000 \
    --end 15000

protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_pair.json \
    --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
    --seeds 101 \
    --use_msa_server \
    --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/pair_5recycle/sample_3w_pair_4.lmdb \
    --start 15000 \
    --end 20000

protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_pair.json \
    --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
    --seeds 101 \
    --use_msa_server \
    --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/pair_5recycle/sample_3w_pair_5.lmdb \
    --start 20000 \
    --end 25000

protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_pair.json \
    --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
    --seeds 101 \
    --use_msa_server \
    --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/pair_5recycle/sample_3w_pair_6.lmdb \
    --start 25000 \
    --end 30000

protenix predict --input /data/rerank/protenix/chembl_bdb/sample_3w/sample_3w_pair.json \
    --out_dir /data/rerank/protenix/chembl_bdb/sample_3w/output \
    --seeds 101 \
    --use_msa_server \
    --lmdb /data/rerank/protenix/chembl_bdb/sample_3w/pair_5recycle/sample_3w_pair_7.lmdb \
    --start 30000