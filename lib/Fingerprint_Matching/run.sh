python3 train.py --batch-size 16 --epochs 30 --lr 0.01 --model-name "TESTBS16LR01"

---CCR
python3 train.py --batch-size 64 --epochs 50 --lr 0.01 --model-name "BS64LR01" --ccr

python3 train.py --batch-size 64 --epochs 50 --lr 0.001 --model-name "NUMPOS3_BS64LR01" --ccr

python3 train.py --batch-size 64 --epochs 50 --lr 0.001 --model-name "LP2_NUMPOS3_BS64LR01" --ccr
python3 train.py --batch-size 64 --epochs 50 --lr 0.001 --model-name "NoTransform_LP2_NUMPOS3_BS64LR01" --ccr

python3 train_DA.py --batch-size 64 --epochs 50 --lr 0.001 --model-name "Train_DA"

cpn-h23-[34,36]