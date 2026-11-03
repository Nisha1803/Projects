#!/bin/bash
set -euo pipefail

base="/Users/ethandoherty/Library/CloudStorage/Box-Box/RCS_Project/RCS_Rework19HM"

proc_dir="${base}"
job_dir="${base}"

n=$(ls "${proc_dir}"/RCS_CA_Procedure_*.R 2>/dev/null | wc -l | tr -d ' ')
if [[ "${n}" -lt 1 ]]; then
  echo "ERROR: No procedure files found in ${proc_dir}"
  exit 1
fi

echo "Found ${n} procedure files."

# IMPORTANT: iterate in natural numeric order (001, 002, ..., 010, 011, ...)
while IFS= read -r f
do
  # Extract numeric id (strip leading zeros safely)
  raw=$(basename "$f" | sed -E 's/^RCS_CA_Procedure_([0-9]+)\.R$/\1/')
  id=$((10#$raw))                 # 010 -> 10, 001 -> 1, 1000 -> 1000
  pad=$(printf "%03d" "$id")      # padded form ONLY for job naming/logs

  jobfile="${job_dir}/FullSimulationLarge_${pad}.sh"

  cat <<EOF > "${jobfile}"
#!/bin/bash

#SBATCH --job-name=HMSim_${pad}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=Sim_${pad}.%j.out
#SBATCH --error=Sim_${pad}.%j.err

module load proj/9.2.1 gdal/3.8.3 r/4.4.0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Starting HM simulation ${pad} with \${SLURM_CPUS_PER_TASK} cores…"

# Run the EXACT procedure file that this loop matched
srun /usr/bin/time -v Rscript "$(basename "$f")"
EOF

  chmod +x "${jobfile}"
done < <(ls -1 "${proc_dir}"/RCS_CA_Procedure_*.R 2>/dev/null | sort -V)

echo "All ${n} job files created in project root:"
echo "  ${job_dir}"
