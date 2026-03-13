#!/bin/bash
#PBS -N ZIP_DATA
#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=4:mem=100gb:scratch_local=50gb

#PBS -m ae
#PBS -j oe

set -e

# Ensure clean_scratch runs on exit, even on error
cleanup() {
echo "Running clean_scratch at $(date)"
clean_scratch
}
trap cleanup EXIT

cd "$SCRATCHDIR"


# Script to create a zip file from all files with the same prefix from subdirectories

PROJECT_NAME="psilocybin-eeg"
SERVER_LOCATION="praha1"
DATADIR="/storage/$SERVER_LOCATION/home/$USER/$PROJECT_NAME"

# User needs to specify input in qsub command like:
#       `qsub -v input={input_value} {pbs_script}`
FILENAME="$input"
SEARCH_DIR="$DATADIR/data/processed/psilo_music"

# Extract the prefix (filename without extension)
# This removes the last dot and everything after it
# PREFIX="${FILENAME%.*}"
PREFIX=$FILENAME

# Set output zip name (use provided name or default to prefix.zip)
OUTPUT_ZIP="${FILENAME}.zip"

# Check if search directory exists
if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Search directory '$SEARCH_DIR' does not exist"
    exit 1
fi

# Check if zip command is available
if ! command -v zip &> /dev/null; then
    echo "Error: 'zip' command not found. Please install zip utility."
    exit 1
fi

echo "Searching for files with prefix: '$PREFIX'"
echo "Search directory: $SEARCH_DIR"
echo "Output zip file: $OUTPUT_ZIP"
echo "---"

# Create a temporary file to store the list of files
TEMP_FILE=$(mktemp)

# Find all files that start with the prefix
find "$SEARCH_DIR" -type f -name "${PREFIX}*" -print0 > "$TEMP_FILE"

# Count files found
COUNT=$(tr -cd '\0' < "$TEMP_FILE" | wc -c)

if [ "$COUNT" -eq 0 ]; then
    echo "No files found with prefix '$PREFIX'"
    rm "$TEMP_FILE"
    exit 0
fi

echo "Found $COUNT file(s)"
echo "---"

# Remove existing zip file if it exists
[ -f "$OUTPUT_ZIP" ] && rm "$OUTPUT_ZIP"

# Create zip file from the found files
# Read each file and add it to the zip
while IFS= read -r -d '' file; do
    echo "Adding: $file"
    zip -q "$OUTPUT_ZIP" "$file"
done < "$TEMP_FILE"

# Clean up temporary file
rm "$TEMP_FILE"

echo "---"
if [ -f "$OUTPUT_ZIP" ]; then
    ZIP_SIZE=$(du -h "$OUTPUT_ZIP" | cut -f1)
    echo "Successfully created: $OUTPUT_ZIP (Size: $ZIP_SIZE)"
    echo "Total files archived: $COUNT"
else
    echo "Error: Failed to create zip file"
    exit 1
fi

mv $OUTPUT_ZIP "$DATADIR/zip_files/$OUTPUT_ZIP"

clean_scratch
