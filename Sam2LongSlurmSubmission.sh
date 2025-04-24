#!/bin/bash

# SAM2Long SLURM job submission script

# Set default configuration parameters (can be modified as needed)
VIDEO_PATH=""
POINTS=""
CHECKPOINT="tiny"
OUTPUT_TYPE="check"
JOB_NAME="sam2long"
LOG_DIR="/home/fortson/ribei056/data/Leech"
OUTDIRECTORY=$LOG_DIR
PYTHON_PATH="python"  # Adjust if using specific environment

# Create logs directory if it doesn't exist
mkdir -p $LOG_DIR
mkdir -p $OUTDIRECTORY

# Generate a timestamp for unique job ID
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to display usage information
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --video PATH       Path to input video (required)"
    echo "  --points POINTS    Points as 'x1,y1:1;x2,y2:0' where 1=include, 0=exclude"
    echo "  --checkpoint TYPE  Model checkpoint: tiny, small, base-plus (default: tiny)"
    echo "  --output TYPE      Output type: check or render (default: check)"
    echo "  --job-name NAME    SLURM job name (default: sam2long)"
    echo "  --help             Display this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        --points)
            POINTS="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output)
            OUTPUT_TYPE="$2"
            shift 2
            ;;
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Verify required arguments
if [ -z "$VIDEO_PATH" ]; then
    echo "Error: --video argument is required"
    usage
    exit 1
fi

# Create SLURM submission file
SLURM_SCRIPT="${JOB_NAME}_${TIMESTAMP}.slurm"

cat > $SLURM_SCRIPT << EOF
#!/bin/bash -l
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_%j.err
#SBATCH --partition=a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH --tmp=32g
#SBATCH --ntasks=8
#SBATCH --time=3:00:00

# Load any required modules or activate virtual environment here
module load conda
conda activate samlong

# Print job information
echo "Job started at \$(date)"
echo "Running on node: \$(hostname)"
echo "CUDA devices: \$CUDA_VISIBLE_DEVICES"
nvidia-smi

# Define script parameters
VIDEO_PATH="${VIDEO_PATH}"
POINTS="${POINTS}"
CHECKPOINT="${CHECKPOINT}"
OUTPUT_TYPE="${OUTPUT_TYPE}"
OUTDIRECTORY="${OUTDIRECTORY}"

# Execute the Python script with parameters
${PYTHON_PATH} /home/fortson/ribei056/software/python/SAM2Long/sam2long_runner.py \\
    --video "\$VIDEO_PATH" \\
    --checkpoint "\$CHECKPOINT" \\
    --output "\$OUTPUT_TYPE" \\
    --outdir "\$OUTDIRECTORY" \\
    $([ ! -z "$POINTS" ] && echo "--points \"\$POINTS\"")

echo "Job completed at \$(date)"
EOF

chmod +x $SLURM_SCRIPT

echo "Created SLURM submission script: $SLURM_SCRIPT"
echo "Submit job with: sbatch $SLURM_SCRIPT"

# Optionally submit the job
read -p "Do you want to submit the job now? (y/n): " SUBMIT
if [[ "$SUBMIT" =~ [yY] ]]; then
    sbatch $SLURM_SCRIPT
    echo "Job submitted!"
fi
