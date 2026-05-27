# # base adam 1e-3
# python train.py name=base

# python train.py name=lr_2e4 \
#     optim=adam \
#     optim.lr=2e-4

# python train.py name=lr_1e4 \
#     optim=adam \
#     optim.lr=1e-4

# python train.py name=lr_2e5 \
#     optim=adam \
#     optim.lr=2e-5

# python train.py name=lr_1e5 \
#     optim=adam \
#     optim.lr=1e-5

# python train.py name=adamw_1e3 \
#     optim=adamw \
#     optim.lr=1e-3

# python train.py name=adamw_2e4 \
#     optim=adamw \
#     optim.lr=2e-4

# python train.py name=adamw_1e4 \
#     optim=adamw \
#     optim.lr=1e-4

# python train.py name=adamw_2e5 \
#     optim=adamw \
#     optim.lr=2e-5

# python train.py name=adamw_1e5 \
#     optim=adamw \
#     optim.lr=1e-5

# precision
# python train.py name=base_tf32 \
#     trainer.max_epochs=15 \

# python train.py name=fp16 \
#     trainer.precision=16 \
#     trainer.max_epochs=15 \

# python train.py name=fp32 \
#     trainer.precision=32 \
#     trainer.max_epochs=15

# python train.py name=bf16 \
#     trainer.precision="bf16" \
#     trainer.max_epochs=15

# python train.py name=16_mixed \
#     trainer.precision="16-mixed" \
#     trainer.max_epochs=15

# python train.py name=bf16_mixed \
#     trainer.precision="bf16-mixed" \
#     trainer.max_epochs=15


# dropout
# python train.py experiment=30w_pair \
#     name=30w_dropout_0.1 \
#     model.dropout=0.1

# # python train.py experiment=30w_pair \
# #     name=30w_dropout_0.2 \
# #     model.dropout=0.2

# python train.py experiment=30w_pair \
#     name=30w_dropout_0.3 \
#     model.dropout=0.3

# python train.py experiment=30w_pair \
#     name=30w_dropout_0.4 \
#     model.dropout=0.4

# python train.py experiment=30w_pair \
#     name=30w_dropout_0.5 \
#     model.dropout=0.5


# batch size
# python train.py experiment=30w_pair \
#     name=30w_batch_16 \
#     dataset.batch_size=16

# python train.py experiment=30w_pair \
#     name=30w_batch_32 \
#     dataset.batch_size=32

# # python train.py experiment=30w_pair \
# #     name=30w_batch_64 \
# #     dataset.batch_size=64

# python train.py experiment=30w_pair \
#     name=30w_batch_128 \
#     dataset.batch_size=128

# python train.py experiment=30w_pair \
#     name=30w_batch_256 \
#     dataset.batch_size=256

# python train.py experiment=30w_pair \
#     name=30w_batch_512 \
#     dataset.batch_size=512

# python train.py experiment=30w_pair \
#     name=30w_batch_1024 \
#     dataset.batch_size=1024


# # large batchsize lr
# python train.py experiment=30w_pair \
#     name=30w_bz512_lr1e3 \
#     dataset.batch_size=512 \
#     optim.lr=1e-3

# python train.py experiment=30w_pair \
#     name=30w_bz512_lr2e4 \
#     dataset.batch_size=512 \
#     optim.lr=2e-4

# python train.py experiment=30w_pair \
#     name=30w_bz512_lr1e4 \
#     dataset.batch_size=512 \
#     optim.lr=1e-4

# python train.py experiment=30w_pair \
#     name=30w_bz512_lr2e5 \
#     dataset.batch_size=512 \
#     optim.lr=2e-5

# python train.py experiment=30w_pair \
#     name=30w_bz512_lr1e5 \
#     dataset.batch_size=512 \
#     optim.lr=1e-5

# python train.py experiment=30w_pair \
#     name=30w_bz1024_lr1e3 \
#     dataset.batch_size=1024 \
#     optim.lr=1e-3

# python train.py experiment=30w_pair \
#     name=30w_bz1024_lr2e4 \
#     dataset.batch_size=1024 \
#     optim.lr=2e-4

# python train.py experiment=30w_pair \
#     name=30w_bz1024_lr1e4 \
#     dataset.batch_size=1024 \
#     optim.lr=1e-4

# python train.py experiment=30w_pair \
#     name=30w_bz1024_lr2e5 \
#     dataset.batch_size=1024 \
#     optim.lr=2e-5

# python train.py experiment=30w_pair \
#     name=30w_bz1024_lr1e5 \
#     dataset.batch_size=1024 \
#     optim.lr=1e-5

# *
# python train.py name=3w_base

# # *
# python train.py name=30w_base \
#     experiment=30w_pair

# # Train commands for all listed strategies

# python train.py experiment=30w_pair \
#     model.strategy="s_input" \
#     name=30w_s_input

# python train.py experiment=30w_pair \
#     model.strategy="s_input_p_only" \
#     name=30w_s_input_p_only

# python train.py experiment=30w_pair \
#     model.strategy="s_input_m_only" \
#     name=30w_s_input_m_only

# python train.py experiment=30w_pair \
#     model.strategy="s" \
#     name=30w_s

# python train.py experiment=30w_pair \
#     model.strategy="s_p_only" \
#     name=30w_s_p_only

# python train.py experiment=30w_pair \
#     model.strategy="s_m_only" \
#     name=30w_s_m_only

# python train.py experiment=30w_pair \
#     model.strategy="z" \
#     name=30w_z

# python train.py experiment=30w_pair \
#     model.strategy="zdouble" \
#     name=30w_zdouble

# python train.py experiment=30w_pair \
#     model.strategy="cat_sz" \
#     name=30w_cat_sz

# python train.py experiment=30w_pair \
#     model.strategy="cat_s_zdouble" \
#     name=30w_cat_s_zdouble

# python train.py experiment=30w_pair \
#     model.strategy="cat_s_input_z" \
#     name=30w_cat_s_input_z

# python train.py experiment=30w_pair \
#     model.strategy="cat_s_input_zdouble" \
#     name=30w_cat_s_input_zdouble

# python train.py experiment=30w_pair \
#     model.strategy="cat_s_input_s" \
#     name=30w_cat_s_input_s

# python train.py experiment=30w_pair \
#     model.strategy="cat_s_input_s_z" \
#     name=30w_cat_s_input_s_z

# python train.py experiment=30w_pair \
#     model.strategy="cat_s_input_s_zdouble" \
#     name=30w_cat_s_input_s_zdouble


# python train.py experiment=30w_pair \
#     model.strategy="cat_s_m_only_z" \
#     name=30w_cat_s_m_only_z

# python train.py experiment=30w_pair_rerank \
#     name=rerank_no_screen

# python train.py experiment=30w_pair_rerank \
#     name=rerank_3w_screen \
#     dataset.dataset_args.train.include_screen=True \
#     dataset.dataset_args.train.max_screen=30000

# python train.py experiment=30w_pair_rerank \
#     name=rerank_6w_screen \
#     dataset.dataset_args.train.include_screen=True \
#     dataset.dataset_args.train.max_screen=60000

# python train.py experiment=30w_pair_rerank \
#     name=rerank_9w_screen \
#     dataset.dataset_args.train.include_screen=True \
#     dataset.dataset_args.train.max_screen=90000

# python train.py experiment=30w_pair_rerank \
#     name=rerank_12w_screen \
#     dataset.dataset_args.train.include_screen=True \
#     dataset.dataset_args.train.max_screen=120000

# python train.py experiment=30w_pair_rerank \
#     name=rerank_15w_screen \
#     dataset.dataset_args.train.include_screen=True \
#     dataset.dataset_args.train.max_screen=150000

# python train.py experiment=30w_pair_screen \
#     name=screen_no_rerank

# python train.py experiment=30w_pair_screen \
#     name=screen_3w_rerank \
#     dataset.dataset_args.train.include_rerank=True \
#     dataset.dataset_args.train.max_rerank=30000

# python train.py experiment=30w_pair_screen \
#     name=screen_6w_rerank \
#     dataset.dataset_args.train.include_rerank=True \
#     dataset.dataset_args.train.max_rerank=60000

# python train.py experiment=30w_pair_screen \
#     name=screen_9w_rerank \
#     dataset.dataset_args.train.include_rerank=True \
#     dataset.dataset_args.train.max_rerank=90000

# python train.py experiment=30w_pair_screen \
#     name=screen_12w_rerank \
#     dataset.dataset_args.train.include_rerank=True \
#     dataset.dataset_args.train.max_rerank=120000

# python train.py experiment=30w_pair_screen \
#     name=screen_15w_rerank \
#     dataset.dataset_args.train.include_rerank=True \
#     dataset.dataset_args.train.max_rerank=150000

# linear probe

# python train.py experiment=30w_pair_linear \
#     model.strategy="s_input" \
#     name=30w_linear_s_input

# python train.py experiment=30w_pair_linear \
#     model.strategy="s_input_p_only" \
#     name=30w_linear_s_input_p_only

# python train.py experiment=30w_pair_linear \
#     model.strategy="s_input_m_only" \
#     name=30w_linear_s_input_m_only

# python train.py experiment=30w_pair_linear \
#     model.strategy="s" \
#     name=30w_linear_s

# python train.py experiment=30w_pair_linear \
#     model.strategy="s_p_only" \
#     name=30w_linear_s_p_only

# python train.py experiment=30w_pair_linear \
#     model.strategy="s_m_only" \
#     name=30w_linear_s_m_only

# python train.py experiment=30w_pair_linear \
#     model.strategy="z" \
#     name=30w_linear_z

# python train.py experiment=30w_pair_linear \
#     model.strategy="zdouble" \
#     name=30w_linear_zdouble

# python train.py experiment=30w_pair_linear \
#     model.strategy="cat_sz" \
#     name=30w_linear_cat_sz

# python train.py experiment=30w_pair_linear \
#     model.strategy="cat_s_zdouble" \
#     name=30w_linear_cat_s_zdouble

# python train.py experiment=30w_pair_linear \
#     model.strategy="cat_s_input_z" \
#     name=30w_linear_cat_s_input_z

# python train.py experiment=30w_pair_linear \
#     model.strategy="cat_s_input_zdouble" \
#     name=30w_linear_cat_s_input_zdouble

# python train.py experiment=30w_pair_linear \
#     model.strategy="cat_s_input_s" \
#     name=30w_linear_cat_s_input_s

# python train.py experiment=30w_pair_linear \
#     model.strategy="cat_s_input_s_z" \
#     name=30w_linear_cat_s_input_s_z

# python train.py experiment=30w_pair_linear \
#     model.strategy="cat_s_input_s_zdouble" \
#     name=30w_linear_cat_s_input_s_zdouble

# python train.py experiment=30w_pair_linear \
#     model.strategy="cat_s_m_only_z" \
#     name=30w_linear_cat_s_m_only_z

# 50epoch rerank
# python train.py experiment=30w_pair_rerank \
#     name=rerank_no_screen_50epoch \
#     trainer.max_epochs=50

# triplet

# python train.py experiment=30w_triplet \
#     model.strategy="s_input" \
#     name=30w_triplet_s_input

# python train.py experiment=30w_triplet \
#     model.strategy="s_input_p_only" \
#     name=30w_triplet_s_input_p_only

# python train.py experiment=30w_triplet \
#     model.strategy="s_input_m1_only" \
#     name=30w_triplet_s_input_m1_only

# python train.py experiment=30w_triplet \
#     model.strategy="s_input_m2_only" \
#     name=30w_triplet_s_input_m2_only

# python train.py experiment=30w_triplet \
#     model.strategy="s_input_m1_m2" \
#     name=30w_triplet_s_input_m1_m2

# python train.py experiment=30w_triplet \
#     model.strategy="s" \
#     name=30w_triplet_s

# python train.py experiment=30w_triplet \
#     model.strategy="s_p_only" \
#     name=30w_triplet_s_p_only

# python train.py experiment=30w_triplet \
#     model.strategy="s_m1_only" \
#     name=30w_triplet_s_m1_only

# python train.py experiment=30w_triplet \
#     model.strategy="s_m2_only" \
#     name=30w_triplet_s_m2_only

# python train.py experiment=30w_triplet \
#     model.strategy="s_m1_m2" \
#     name=30w_triplet_s_m1_m2

# python train.py experiment=30w_triplet \
#     model.strategy="z_pm1_pm2" \
#     name=30w_triplet_z_pm1_pm2

# python train.py experiment=30w_triplet \
#     model.strategy="z_pm1_only" \
#     name=30w_triplet_z_pm1_only

# python train.py experiment=30w_triplet \
#     model.strategy="z_pm2_only" \
#     name=30w_triplet_z_pm2_only

# python train.py experiment=30w_triplet \
#     model.strategy="cat_s_z_pm1_pm2" \
#     name=30w_triplet_cat_s_z_pm1_pm2

# python train.py experiment=30w_triplet \
#     model.strategy="cat_s_m1_m2_z_pm1_pm2" \
#     name=30w_triplet_cat_s_m1_m2_z_pm1_pm2

# python train.py experiment=30w_triplet \
#     model.strategy="cat_s_input_m1_m2_s_m1_m2_z_pm1_pm2" \
#     name=30w_triplet_cat_s_input_m1_m2_s_m1_m2_z_pm1_pm2

# # linear probe 
# python train.py experiment=30w_triplet_linear \
#     model.strategy="s_input" \
#     name=30w_triplet_linear_s_input

# python train.py experiment=30w_triplet_linear \
#     model.strategy="s_input_p_only" \
#     name=30w_triplet_linear_s_input_p_only

# python train.py experiment=30w_triplet_linear \
#     model.strategy="s_input_m1_only" \
#     name=30w_triplet_linear_s_input_m1_only

# python train.py experiment=30w_triplet_linear \
#     model.strategy="s_input_m2_only" \
#     name=30w_triplet_linear_s_input_m2_only

# python train.py experiment=30w_triplet_linear \
#     model.strategy="s_input_m1_m2" \
#     name=30w_triplet_linear_s_input_m1_m2

# python train.py experiment=30w_triplet_linear \
#     model.strategy="s" \
#     name=30w_triplet_linear_s

# python train.py experiment=30w_triplet_linear \
#     model.strategy="s_p_only" \
#     name=30w_triplet_linear_s_p_only

# python train.py experiment=30w_triplet_linear \
#     model.strategy="s_m1_only" \
#     name=30w_triplet_linear_s_m1_only

# python train.py experiment=30w_triplet_linear \
#     model.strategy="s_m2_only" \
#     name=30w_triplet_linear_s_m2_only

# python train.py experiment=30w_triplet_linear \
#     model.strategy="s_m1_m2" \
#     name=30w_triplet_linear_s_m1_m2

# python train.py experiment=30w_triplet_linear \
#     model.strategy="z_pm1_pm2" \
#     name=30w_triplet_linear_z_pm1_pm2

# python train.py experiment=30w_triplet_linear \
#     model.strategy="z_pm1_only" \
#     name=30w_triplet_linear_z_pm1_only

# python train.py experiment=30w_triplet_linear \
#     model.strategy="z_pm2_only" \
#     name=30w_triplet_linear_z_pm2_only

# python train.py experiment=30w_triplet_linear \
#     model.strategy="cat_s_z_pm1_pm2" \
#     name=30w_triplet_linear_cat_s_z_pm1_pm2

# python train.py experiment=30w_triplet_linear \
#     model.strategy="cat_s_m1_m2_z_pm1_pm2" \
#     name=30w_triplet_linear_cat_s_m1_m2_z_pm1_pm2

# python train.py experiment=30w_triplet_linear \
#     model.strategy="cat_s_input_m1_m2_s_m1_m2_z_pm1_pm2" \
#     name=30w_triplet_linear_cat_s_input_m1_m2_s_m1_m2_z_pm1_pm2

# diff

# python train.py experiment=30w_triplet \
#     model.strategy="diff_s_input" \
#     name=30w_triplet_diff_s_input

# python train.py experiment=30w_triplet \
#     model.strategy="diff_s" \
#     name=30w_triplet_diff_s

# python train.py experiment=30w_triplet \
#     model.strategy="diff_z" \
#     name=30w_triplet_diff_z

# python train.py experiment=30w_triplet \
#     model.strategy="diff_s_z" \
#     name=30w_triplet_diff_s_z

# python train.py experiment=30w_triplet \
#     model.strategy="diff_cat_s_input" \
#     name=30w_triplet_diff_cat_s_input

# python train.py experiment=30w_triplet \
#     model.strategy="diff_cat_s" \
#     name=30w_triplet_diff_cat_s

# python train.py experiment=30w_triplet \    
#     model.strategy="diff_cat_z" \
#     name=30w_triplet_diff_cat_z

# python train.py experiment=30w_triplet \
#     model.strategy="diff_cat_s_z" \
#     name=30w_triplet_diff_cat_s_z

# screen / rerank

# python train.py experiment=30w_triplet_rerank \
#     name=triplet_rerank_no_screen

# python train.py experiment=30w_triplet_rerank \
#     name=triplet_rerank_3w_screen \
#     dataset.dataset_args.train.include_screen=True \
#     dataset.dataset_args.train.max_screen=30000

# python train.py experiment=30w_triplet_rerank \
#     name=triplet_rerank_6w_screen \
#     dataset.dataset_args.train.include_screen=True \
#     dataset.dataset_args.train.max_screen=60000

# python train.py experiment=30w_triplet_rerank \
#     name=triplet_rerank_9w_screen \
#     dataset.dataset_args.train.include_screen=True \
#     dataset.dataset_args.train.max_screen=90000

# python train.py experiment=30w_triplet_rerank \
#     name=triplet_rerank_12w_screen \
#     dataset.dataset_args.train.include_screen=True \
#     dataset.dataset_args.train.max_screen=120000

# python train.py experiment=30w_triplet_rerank \
#     name=triplet_rerank_15w_screen \
#     dataset.dataset_args.train.include_screen=True \
#     dataset.dataset_args.train.max_screen=150000

# python train.py experiment=30w_triplet_screen \
#     name=triplet_screen_no_rerank

# python train.py experiment=30w_triplet_screen \
#     name=triplet_screen_3w_rerank \
#     dataset.dataset_args.train.include_rerank=True \
#     dataset.dataset_args.train.max_rerank=30000

# python train.py experiment=30w_triplet_screen \
#     name=triplet_screen_6w_rerank \
#     dataset.dataset_args.train.include_rerank=True \
#     dataset.dataset_args.train.max_rerank=60000

# python train.py experiment=30w_triplet_screen \
#     name=triplet_screen_9w_rerank \
#     dataset.dataset_args.train.include_rerank=True \
#     dataset.dataset_args.train.max_rerank=90000

# python train.py experiment=30w_triplet_screen \
#     name=triplet_screen_12w_rerank \
#     dataset.dataset_args.train.include_rerank=True \
#     dataset.dataset_args.train.max_rerank=120000

# python train.py experiment=30w_triplet_screen \
#     name=triplet_screen_15w_rerank \
#     dataset.dataset_args.train.include_rerank=True \
#     dataset.dataset_args.train.max_rerank=150000

# # refine agg
# python train.py experiment=3w_pair_combine \
#     model.n_residue=0 \
#     optim.lr=1e-3 \
#     name=3w_pair_combine_0res_1e3

# python train.py experiment=3w_pair_combine \
#     model.n_residue=0 \
#     optim.lr=2e-4 \
#     name=3w_pair_combine_0res_2e4

# python train.py experiment=3w_pair_combine \
#     model.n_residue=0 \
#     optim.lr=1e-4 \
#     name=3w_pair_combine_0res_1e4

# python train.py experiment=3w_pair_combine \
#     model.n_residue=0 \
#     optim.lr=2e-5 \
#     name=3w_pair_combine_0res_2e5

# python train.py experiment=3w_pair_combine \
#     model.n_residue=1 \
#     optim.lr=1e-3 \
#     name=3w_pair_combine_1res_1e3

# python train.py experiment=3w_pair_combine \
#     model.n_residue=1 \
#     optim.lr=2e-4 \
#     name=3w_pair_combine_1res_2e4

# python train.py experiment=3w_pair_combine \
#     model.n_residue=1 \
#     optim.lr=1e-4 \
#     name=3w_pair_combine_1res_1e4

# python train.py experiment=3w_pair_combine \
#     model.n_residue=1 \
#     optim.lr=2e-5 \
#     name=3w_pair_combine_1res_2e5

# python train.py experiment=3w_pair_combine \
#     model.n_residue=2 \
#     optim.lr=1e-3 \
#     name=3w_pair_combine_2res_1e3

# python train.py experiment=3w_pair_combine \
#     model.n_residue=2 \
#     optim.lr=2e-4 \
#     name=3w_pair_combine_2res_2e4

# python train.py experiment=3w_pair_combine \
#     model.n_residue=2 \
#     optim.lr=1e-4 \
#     name=3w_pair_combine_2res_1e4

# python train.py experiment=3w_pair_combine \
#     model.n_residue=2 \
#     optim.lr=2e-5 \
#     name=3w_pair_combine_2res_2e5


# python train.py experiment=3w_pair_combine_rerank \
#     name=3w_pair_combine_rerank_no_screen

# python train.py experiment=3w_pair_combine_screen \
#     name=3w_pair_combine_screen_no_rerank

# python train.py experiment=3w_pair_combine_rerank_5recycle \
#     name=3w_pair_combine_rerank_no_screen_5recycle

# python train.py experiment=3w_pair_combine_screen_5recycle \
#     name=3w_pair_combine_screen_no_rerank_5recycle

# lr

# python train.py experiment=3w_pair_combine_rerank \
#     optim.lr=1e-4 \
#     name=3w_pair_combine_rerank_no_screen_1e4

# python train.py experiment=3w_pair_combine_screen \
#     optim.lr=1e-4 \
#     name=3w_pair_combine_screen_no_rerank_1e4

# python train.py experiment=3w_pair_combine_rerank_5recycle \
#     optim.lr=1e-4 \
#     name=3w_pair_combine_rerank_no_screen_5recycle_1e4

# python train.py experiment=3w_pair_combine_screen_5recycle \
#     optim.lr=1e-4 \
#     name=3w_pair_combine_screen_no_rerank_5recycle_1e4

# python train.py experiment=3w_pair_combine_rerank \
#     optim.lr=2e-5 \
#     name=3w_pair_combine_rerank_no_screen_2e5

# python train.py experiment=3w_pair_combine_screen \
#     optim.lr=2e-5 \
#     name=3w_pair_combine_screen_no_rerank_2e5

# python train.py experiment=3w_pair_combine_rerank_5recycle \
#     optim.lr=2e-5 \
#     name=3w_pair_combine_rerank_no_screen_5recycle_2e5

# python train.py experiment=3w_pair_combine_screen_5recycle \
#     optim.lr=2e-5 \
#     name=3w_pair_combine_screen_no_rerank_5recycle_2e5

# # from s
# python train.py experiment=3w_pair_combine \
#     model.strategy="from_s" \
#     dataset.batch_size=14 \
#     name=3w_pair_combine_from_s

# python train.py experiment=3w_pair_combine_rerank \
#     model.strategy="from_s" \
#     dataset.batch_size=16 \
#     name=3w_pair_combine_rerank_from_s

# python train.py experiment=3w_pair_combine_screen \
#     model.strategy="from_s" \
#     dataset.batch_size=16 \
#     name=3w_pair_combine_screen_from_s

# # from sz
# python train.py experiment=3w_pair_combine \
#     model.strategy="from_sz" \
#     dataset.batch_size=16 \
#     name=3w_pair_combine_from_sz

# python train.py experiment=3w_pair_combine_rerank \
#     model.strategy="from_sz" \
#     dataset.batch_size=16 \
#     name=3w_pair_combine_rerank_from_sz

# python train.py experiment=3w_pair_combine_screen \
#     model.strategy="from_sz" \
#     dataset.batch_size=16 \
#     name=3w_pair_combine_screen_from_sz

# different identity, triplet, rerank / screen, 11
# python train.py experiment=3w_triplet \
#     name=3w_triplet

# python train.py experiment=3w_triplet \
#     dataset=3w_triplet_60 \
#     name=3w_triplet_60

# python train.py experiment=3w_triplet \
#     dataset=3w_triplet_100 \
#     name=3w_triplet_100

# python train.py experiment=3w_triplet_rerank \
#     name=3w_triplet_rerank

# python train.py experiment=3w_triplet_rerank \
#     dataset=3w_triplet_60 \
#     name=3w_triplet_rerank_60

# python train.py experiment=3w_triplet_rerank \
#     dataset=3w_triplet_100 \
#     name=3w_triplet_rerank_100

# python train.py experiment=3w_triplet_screen \
#     name=3w_triplet_screen

# python train.py experiment=3w_triplet_screen \
#     dataset=3w_triplet_60 \
#     name=3w_triplet_screen_60

# python train.py experiment=3w_triplet_screen \
#     dataset=3w_triplet_100 \
#     name=3w_triplet_screen_100

# different identity, pair, rerank / screen, 18
# python train.py experiment=3w_pair \
#     name=3w_pair

# python train.py experiment=3w_pair \
#     dataset=3w_pair_60 \
#     name=3w_pair_60

# python train.py experiment=3w_pair \
#     dataset=3w_pair_100 \
#     name=3w_pair_100

# python train.py experiment=3w_pair_rerank \
#     name=3w_pair_rerank

# python train.py experiment=3w_pair_rerank \
#     dataset=3w_pair_60 \
#     name=3w_pair_rerank_60

# python train.py experiment=3w_pair_rerank \
#     dataset=3w_pair_100 \
#     name=3w_pair_rerank_100

# python train.py experiment=3w_pair_screen \
#     name=3w_pair_screen

# python train.py experiment=3w_pair_screen \
#     dataset=3w_pair_60 \
#     name=3w_pair_screen_60

# python train.py experiment=3w_pair_screen \
#     dataset=3w_pair_100 \
#     name=3w_pair_screen_100

# 3w triplet

# python train.py experiment=3w_triplet \
#     name=3w_triplet

# python train.py experiment=3w_triplet \
#     dataset=3w_triplet_60 \
#     name=3w_triplet_60

# python train.py experiment=3w_triplet \
#     dataset=3w_triplet_100 \
#     name=3w_triplet_100

# python train.py experiment=3w_triplet_rerank \
#     name=3w_triplet_rerank

# python train.py experiment=3w_triplet_rerank \
#     dataset=3w_triplet_60 \
#     name=3w_triplet_rerank_60

# python train.py experiment=3w_triplet_rerank \
#     dataset=3w_triplet_100 \
#     name=3w_triplet_rerank_100

# # 3w triplet combine
# python train.py experiment=3w_triplet_combine \
#     name=3w_triplet_combine

# python train.py experiment=3w_triplet_combine \
#     dataset=3w_triplet_ori_60 \
#     name=3w_triplet_combine_60

# python train.py experiment=3w_triplet_combine \
#     dataset=3w_triplet_ori_100 \
#     name=3w_triplet_combine_100

# python train.py experiment=3w_triplet_combine_rerank \
#     name=3w_triplet_combine_rerank

# python train.py experiment=3w_triplet_combine_rerank \
#     dataset=3w_triplet_ori_60 \
#     name=3w_triplet_combine_rerank_60

# python train.py experiment=3w_triplet_combine_rerank \
#     dataset=3w_triplet_ori_100 \
#     name=3w_triplet_combine_rerank_100

# # pair
# # python train.py experiment=3w_pair \
# #     name=3w_pair

# python train.py experiment=3w_pair \
#     dataset=3w_pair_60 \
#     name=3w_pair_60

# python train.py experiment=3w_pair \
#     dataset=3w_pair_100 \
#     name=3w_pair_100

# # python train.py experiment=3w_pair_rerank \
# #     name=3w_pair_rerank

# python train.py experiment=3w_pair_rerank \
#     dataset=3w_pair_60 \
#     name=3w_pair_rerank_60

# python train.py experiment=3w_pair_rerank \
#     dataset=3w_pair_100 \
#     name=3w_pair_rerank_100

# # python train.py experiment=3w_pair_screen \
# #     name=3w_pair_screen

# python train.py experiment=3w_pair_screen \
#     dataset=3w_pair_60 \
#     name=3w_pair_screen_60

# python train.py experiment=3w_pair_screen \
#     dataset=3w_pair_100 \
#     name=3w_pair_screen_100

# # 3w pair combine
# # python train.py experiment=3w_pair_combine \
# #     name=3w_pair_combine

# python train.py experiment=3w_pair_combine \
#     dataset=3w_pair_ori_60 \
#     name=3w_pair_combine_60

# python train.py experiment=3w_pair_combine \
#     dataset=3w_pair_ori_100 \
#     name=3w_pair_combine_100

# # python train.py experiment=3w_pair_combine_rerank \
# #     name=3w_pair_combine_rerank

# python train.py experiment=3w_pair_combine_rerank \
#     dataset=3w_pair_ori_60 \
#     name=3w_pair_combine_rerank_60

# python train.py experiment=3w_pair_combine_rerank \
#     dataset=3w_pair_ori_100 \
#     name=3w_pair_combine_rerank_100


# # train 3w_100 pair with 5 random seed
# python train.py experiment=3w_pair_rerank \
#     dataset=3w_pair_100 \
#     trainer.max_epochs=50 \
#     seed=2025 \
#     name=3w_pair_rerank_100_seed_2025

# python train.py experiment=3w_pair_rerank \
#     dataset=3w_pair_100 \
#     trainer.max_epochs=50 \
#     seed=2026 \
#     name=3w_pair_rerank_100_seed_2026

# python train.py experiment=3w_pair_rerank \
#     dataset=3w_pair_100 \
#     trainer.max_epochs=50 \
#     seed=1 \
#     name=3w_pair_rerank_100_seed_1

# python train.py experiment=3w_pair_rerank \
#     dataset=3w_pair_100 \
#     trainer.max_epochs=50 \
#     seed=2 \
#     name=3w_pair_rerank_100_seed_2

# python train.py experiment=3w_pair_rerank \
#     dataset=3w_pair_100 \
#     trainer.max_epochs=50 \
#     seed=7 \
#     name=3w_pair_rerank_100_seed_3

# # train 3w_100 triplet with 5 random seed

# python train.py experiment=3w_triplet_rerank \
#     dataset=3w_triplet_100 \
#     trainer.max_epochs=50 \
#     seed=2025 \
#     name=3w_triplet_rerank_100_seed_2025

# python train.py experiment=3w_triplet_rerank \
#     dataset=3w_triplet_100 \
#     trainer.max_epochs=50 \
#     seed=2026 \
#     name=3w_triplet_rerank_100_seed_2026

# python train.py experiment=3w_triplet_rerank \
#     dataset=3w_triplet_100 \
#     trainer.max_epochs=50 \
#     seed=1 \
#     name=3w_triplet_rerank_100_seed_1

# python train.py experiment=3w_triplet_rerank \
#     dataset=3w_triplet_100 \
#     trainer.max_epochs=50 \
#     seed=2 \
#     name=3w_triplet_rerank_100_seed_2

# python train.py experiment=3w_triplet_rerank \
#     dataset=3w_triplet_100 \
#     trainer.max_epochs=50 \
#     seed=7 \
#     name=3w_triplet_rerank_100_seed_3

# # agg_strategy, triplet
# python train.py experiment=3w_triplet_combine_rerank \
#     dataset=3w_triplet_ori_100 \
#     model.agg_strategy="sum" \
#     name=3w_triplet_combine_rerank_sum

# python train.py experiment=3w_triplet_combine_rerank \
#     dataset=3w_triplet_ori_100 \
#     model.agg_strategy="mean" \
#     name=3w_triplet_combine_rerank_mean

# python train.py experiment=3w_triplet_combine_rerank \
#     dataset=3w_triplet_ori_100 \
#     model.agg_strategy="sum_sigmoid" \
#     name=3w_triplet_combine_rerank_sum_sigmoid

# python train.py experiment=3w_triplet_combine_rerank \
#     dataset=3w_triplet_ori_100 \
#     model.agg_strategy="mean_sigmoid" \
#     name=3w_triplet_combine_rerank_mean_sigmoid

# python train.py experiment=3w_triplet_combine_rerank \
#     dataset=3w_triplet_ori_100 \
#     model.agg_strategy="topk_sum" \
#     name=3w_triplet_combine_rerank_topk_sum

# python train.py experiment=3w_triplet_combine_rerank \
#     dataset=3w_triplet_ori_100 \
#     model.agg_strategy="topk_mean" \
#     name=3w_triplet_combine_rerank_topk_mean

# python train.py experiment=3w_triplet_combine_rerank \
#     dataset=3w_triplet_ori_100 \
#     model.agg_strategy="topk_sum_sigmoid" \
#     name=3w_triplet_combine_rerank_topk_sum_sigmoid

# python train.py experiment=3w_triplet_combine_rerank \
#     dataset=3w_triplet_ori_100 \
#     model.agg_strategy="topk_mean_sigmoid" \
#     name=3w_triplet_combine_rerank_topk_mean_sigmoid

# agg_strategy, pair
# python train.py experiment=3w_pair_combine_rerank \
#     dataset=3w_pair_ori_100 \
#     model.agg_strategy="sum" \
#     name=3w_pair_combine_rerank_sum

# python train.py experiment=3w_pair_combine_rerank \
#     dataset=3w_pair_ori_100 \
#     model.agg_strategy="mean" \
#     name=3w_pair_combine_rerank_mean

# python train.py experiment=3w_pair_combine_rerank \
#     dataset=3w_pair_ori_100 \
#     model.agg_strategy="sum_sigmoid" \
#     name=3w_pair_combine_rerank_sum_sigmoid

# python train.py experiment=3w_pair_combine_rerank \
#     dataset=3w_pair_ori_100 \
#     model.agg_strategy="mean_sigmoid" \
#     name=3w_pair_combine_rerank_mean_sigmoid

# python train.py experiment=3w_pair_combine_rerank \
#     dataset=3w_pair_ori_100 \
#     model.agg_strategy="topk_sum" \
#     name=3w_pair_combine_rerank_topk_sum

# python train.py experiment=3w_pair_combine_rerank \
#     dataset=3w_pair_ori_100 \
#     model.agg_strategy="topk_mean" \
#     name=3w_pair_combine_rerank_topk_mean

# python train.py experiment=3w_pair_combine_rerank \
#     dataset=3w_pair_ori_100 \
#     model.agg_strategy="topk_sum_sigmoid" \
#     name=3w_pair_combine_rerank_topk_sum_sigmoid

# python train.py experiment=3w_pair_combine_rerank \
#     dataset=3w_pair_ori_100 \
#     model.agg_strategy="topk_mean_sigmoid" \
#     name=3w_pair_combine_rerank_topk_mean_sigmoid


# python train.py experiment=3w_pair \
#     dataset=5w_pair_100 \
#     name=5w_pair_100

# python train.py experiment=3w_pair_rerank \
#     dataset=5w_pair_100 \
#     name=5w_pair_100_rerank

# python train.py experiment=3w_triplet \
#     dataset=5w_triplet_100 \
#     name=5w_triplet_100

# python train.py experiment=3w_triplet_rerank \
#     dataset=5w_triplet_100 \
#     name=5w_triplet_100_rerank

# python train.py experiment=3w_pair \
#     dataset=5w_pair_100 \
#     dataset.dataset_args.train.meta_split=train_4w_100 \
#     name=4w_pair_100

# python train.py experiment=3w_pair_rerank \
#     dataset=5w_pair_100 \
#     dataset.dataset_args.train.meta_split=train_4w_100 \
#     name=4w_pair_100_rerank

# python train.py experiment=3w_triplet \
#     dataset=5w_triplet_100 \
#     dataset.dataset_args.train.meta_split=train_4w_100 \
#     name=4w_triplet_100

# python train.py experiment=3w_triplet_rerank \
#     dataset=5w_triplet_100 \
#     dataset.dataset_args.train.meta_split=train_4w_100 \
#     name=4w_triplet_100_rerank

# python train.py experiment=3w_pair \
#     dataset=5w_pair_100 \
#     dataset.dataset_args.train.meta_split=train_3.5w_100 \
#     name=3.5w_pair_100

# python train.py experiment=3w_pair_rerank \
#     dataset=5w_pair_100 \
#     dataset.dataset_args.train.meta_split=train_3.5w_100 \
#     name=3.5w_pair_100_rerank

# python train.py experiment=3w_triplet \
#     dataset=5w_triplet_100 \
#     dataset.dataset_args.train.meta_split=train_3.5w_100 \
#     name=3.5w_triplet_100

# python train.py experiment=3w_triplet_rerank \
#     dataset=5w_triplet_100 \
#     dataset.dataset_args.train.meta_split=train_3.5w_100 \
#     name=3.5w_triplet_100_rerank

# python train.py experiment=3w_pair_rerank \
#     dataset=5w_pair_100 \
#     trainer.max_epochs=50 \
#     name=5w_pair_100_rerank_50epoch

# python train.py experiment=3w_pair_rerank \
#     dataset=5w_pair_100 \
#     trainer.max_epochs=50 \
#     scheduler=cosine_annealing \
#     name=5w_pair_100_rerank_50epoch_lrdecay

# python train.py experiment=3w_pair_rerank \
#     dataset=5w_pair_100 \
#     trainer.max_epochs=50 \
#     scheduler=cosine_annealing \
#     scheduler.num_warmup_steps=200 \
#     name=5w_pair_100_rerank_50epoch_lrdecay_warmup

python train.py experiment=3w_pair_rerank \
    dataset=test3 \
    trainer.max_epochs=60 \
    name=5w_pair_100_rerank_50epoch \
    +trainer.limit_val_batches=0.0
