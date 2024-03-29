#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=Road_Damage_Detection
#SBATCH --output=Road_Damage_Detection.out
#SBATCH --error=Road_Damage_Detection.err
#SBATCH --partition=CPUQ
#SBATCH --gres=gpu:1
#SBATCH --account=iv-imt
#SBATCH --nodes=1
#SBATCH --mail-user=mdaaz@stud.ntnu.no


python scripts/train.py --root_dir /cluster/projects/vc/courses/TDT17/2022/open/RDD2022/Norway --model_name fasterrcnn_resnet50 --num_epochs 100
