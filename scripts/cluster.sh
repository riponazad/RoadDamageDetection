#!/bin/bash

#SBATCH --time=12:01:00
#SBATCH --job-name=Road_Damage_Detection
#SBATCH --output=Road_Damage_Detection.out
#SBATCH --error=Road_Damage_Detection.err
#SBATCH --partition=CPUQ
#SBATCH --gres=gpu:1
#SBATCH --account=mdaaz
#SBATCH --nodes=1
#SBATCH --mail-user=mdaaz@stud.ntnu.no


echo $HOSTNAME