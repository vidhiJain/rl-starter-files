#!/bin/sh 
ENV="DistShift1pposhapcoeff"
names= find storage/ -name "$ENV*"
echo $names
EPISODES=1000
SEED=42

for MODEL in  DistShift1pposhapcoeff100 DistShift1pposhapcoeff0.1 DistShift1pposhapcoeff50 DistShift1pposhapcoeff-1 DistShift1pposhapcoeff2 DistShift1pposhapcoeff10 DistShift1pposhapcoeff0 DistShift1pposhapcoeff20
do 
    echo $MODEL
    python -m scripts.evaluate --env MiniGrid-DistShift2-v0 --model $MODEL --episodes $EPISODES --seed $SEED
done

echo "Run \npython plot_train_returns.py"