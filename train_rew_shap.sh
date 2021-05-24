ENV="MiniGrid-DistShift1-v0"
MODEL="DistShiftrewL1coeff"
GPU=5

for COEFF in 0 1 10
do
    echo $COEFF
    CUDA_VISIBLE_DEVICES=$GPU python -m scripts.train --env $ENV \
     --model $MODEL$COEFF \
     --algo pposhapl1rew --shap_coeff 0 --frames 80000
done