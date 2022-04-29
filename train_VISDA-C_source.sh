PORT=10000
MEMO="source"

for SEED in 2020 2021 2022
do
    python main_adacontrast.py train_source=true learn=source \
    seed=${SEED} port=${PORT} memo=${MEMO} project="VISDA-C" \
    data.data_root="${PWD}/datasets" data.workers=8 \
    data.dataset="VISDA-C" data.source_domains="[train]" data.target_domains="[validation]" \
    learn.epochs=10 \
    model_src.arch="resnet101" \
    optim.lr=2e-4 
done