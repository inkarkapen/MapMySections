#!/bin/bash
#SBATCH --job-name=model_train_merfish_and_scans_3D_gpu_    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=inkar.kapen@alleninstitute.org     # Where to send mail  
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --mem=256gb                     # Job memory request (per node)
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=logs/model_train_merfish_and_scans_3D_gpu_%j.log   # Standard output and error log
#SBATCH --partition celltypes         # Partition used for processing
#SBATCH --tmp=30G                     # Request the amount of space your jobs needs on /scratch/fast
#SBATCH --cpus-per-task=32             # Number of CPUs per task
#SBATCH --gres gpu:1

# --gres gpu:1

pwd; hostname; date
 
echo "Running a model training"

source ~/.bash_profile

#source /allen/programs/celltypes/workgroups/rnaseqanalysis/EvoGen/Team/Inkar/2025_Hackathon/MMC/.venv/bin/activate

source activate /allen/programs/celltypes/workgroups/rnaseqanalysis/EvoGen/Team/Inkar/miniconda3/envs/abc_project

python model_train_merfish.py
python model_train_scans.py

echo "Finished running a python script on a single thread"
date

