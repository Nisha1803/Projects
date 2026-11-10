#!/bin/bash
#SBATCH --job-name=RCS_ABC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=RCS_ABC.%j.out
#SBATCH --error=RCS_ABC.%j.err
# Submit from rcs_abc/:  sbatch scripts/run_abc_ethan.slurm /path/to/ethan_project
# One simulation is ~13 s and well under 2 GB, so 32 cores x 2 GB is ample.
module load proj/9.2.1 gdal/3.8.3 r/4.4.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ABC_CORES=${SLURM_CPUS_PER_TASK}
Rscript scripts/run_abc_ethan.R "$1"
