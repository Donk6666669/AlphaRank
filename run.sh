python scripts/preprocess/json2feature.py \
    --json_path /data/rerank/protenix/chembl_bdb/chembl_bdb_protenix.json \
    --lmdb_path /data/rerank/protenix/chembl_bdb/chembl_bdb.lmdb


 python scripts/colabfold_msa.py \
    /data_hdd/protein/rerank/protenix/chembl_bdb/chembl_bdb_unique_sequences.fasta \
    /data_ssd/protein/AIRFold/datasets_2024/database \
    /data_hdd/protein/rerank/protenix/chembl_bdb/msa/data \
    --db1 uniref30_2302_db \
    --db3 colabfold_envdb_202108_db \
    --mmseqs /data_hdd/home/casp15/code/mmseqs_test//MMseqs2/build/bin/mmseqs \
    --db_load_mode 2