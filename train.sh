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

# *
# python train.py name=30w_base \
#     experiment=30w_pair