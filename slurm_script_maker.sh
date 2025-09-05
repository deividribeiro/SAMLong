#!/bin/bash

# SAM2Long SLURM job submission script

# Set default configuration parameters (can be modified as needed)
VIDEO_PATH=""
POINTS=""
CHECKPOINT="tiny"
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
    echo "  --frame  FRAME     Index number for starting frame "
    echo "  --checkpoint TYPE  Model checkpoint: tiny, small, base-plus (default: tiny)"
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
        --job-name)
            JOB_NAME="$2"
            shift 2
            ;;
        --frame)
            FRAME="$2"
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
#SBATCH --ntasks=4
#SBATCH --time=6:00:00

# Load any required modules or activate virtual environment here
module load conda
conda activate samlong

# Print job information
echo "Job started at \$(date)"
echo "Running on node: \$(hostname)"
echo "CUDA devices: \$CUDA_VISIBLE_DEVICES"
nvidia-smi

# Define script parameters
VIDEO_PATH="$(echo ${VIDEO_PATH} | sed 's/\\/\\\\/g; s/"/\\"/g')"

# Now process the video path
SAM2LONG_DIR="/home/fortson/ribei056/software/python/SAM2Long"
LOG_DIR=${LOG_DIR}
VIDEO_PATH_BASE="$(basename "${VIDEO_PATH}")"
OLD_VIDEO_PATH_DIR="$(dirname "${VIDEO_PATH}")"
VIDEO_PATH_DIR="\$(basename "\${OLD_VIDEO_PATH_DIR}")"
mkdir -p "\${LOG_DIR}/\${VIDEO_PATH_DIR}"
cp "\${VIDEO_PATH}" "\${LOG_DIR}/\${VIDEO_PATH_DIR}/\${VIDEO_PATH_BASE}"
ffmpeg -i "\${LOG_DIR}/\${VIDEO_PATH_DIR}/\${VIDEO_PATH_BASE}" "\${LOG_DIR}/\${VIDEO_PATH_DIR}/\${VIDEO_PATH_BASE%.*}.mp4"
rm "\${LOG_DIR}/\${VIDEO_PATH_DIR}/\${VIDEO_PATH_BASE}"
VIDEO_PATH="\${LOG_DIR}/\${VIDEO_PATH_DIR}/\${VIDEO_PATH_BASE%.*}.mp4"

POINTS="${POINTS}"
CHECKPOINT="${CHECKPOINT}"
OUTDIRECTORY="${OUTDIRECTORY}"
FRAME="${FRAME}"

# Execute the Python script with parameters
${PYTHON_PATH} ${SAM2LONG_DIR}/sam2long_processor.py \\
    --video "\$VIDEO_PATH" \\
    --checkpoint "\$CHECKPOINT" \\
    --outdir "\$OUTDIRECTORY" \\
    $([ ! -z "$FRAME" ] && echo "--frame \"\$FRAME\"") \\
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
