#!/bin/sh
python -m scripts.evaluate --env MiniGrid-DoorKey-8x8-v0 --model DoorKeyppoonly --episodes 10 --seed 42 
python -m scripts.evaluate --env MiniGrid-DoorKey-8x8-v0 --model DoorKeypposhapcoeffcommented --episodes 10 --seed 42 
python -m scripts.evaluate --env MiniGrid-DoorKey-8x8-v0 --model DoorKeypposhapcoeff0 --episodes 10 --seed 42 
python -m scripts.evaluate --env MiniGrid-DoorKey-8x8-v0 --model DoorKeypposhapcoeff0.1 --episodes 10 --seed 42
python -m scripts.evaluate --env MiniGrid-DoorKey-8x8-v0 --model DoorKeypposhapcoeff0.5 --episodes 10 --seed 42
python -m scripts.evaluate --env MiniGrid-DoorKey-8x8-v0 --model DoorKeypposhapcoeff0.5 --episodes 10 --seed 42
python -m scripts.evaluate --env MiniGrid-DoorKey-8x8-v0 --model DoorKeypposhapcoeff0.7 --episodes 10 --seed 42
python -m scripts.evaluate --env MiniGrid-DoorKey-8x8-v0 --model DoorKeypposhapcoeff1 --episodes 10 --seed 42
python -m scripts.evaluate --env MiniGrid-DoorKey-8x8-v0 --model DoorKeypposhapcoeff10 --episodes 10 --seed 42
python -m scripts.evaluate --env MiniGrid-DoorKey-8x8-v0 --model DoorKeypposhapcoeff20 --episodes 10 --seed 42
python -m scripts.evaluate --env MiniGrid-DoorKey-8x8-v0 --model DoorKeypposhapcoeff100 --episodes 10 --seed 42