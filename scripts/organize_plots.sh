#!/bin/bash

# Script to organize files by PSI{number} prefix into subdirectories
# Usage: ./organize_psi_files.sh <source_directory> <destination_directory>

# Check if correct number of arguments provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <source_directory>
    echo "Example: $0 /path/to/files
    exit 1
fi

SOURCE_DIR="$1"
DEST_DIR="$SOURCE_DIR"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist"
    exit 1
fi

# Create destination directory if it doesn't exist
if [ ! -d "$DEST_DIR" ]; then
    echo "Creating destination directory: $DEST_DIR"
    mkdir -p "$DEST_DIR"
fi

echo "Organizing files from: $SOURCE_DIR"
echo "Destination: $DEST_DIR"
echo "---"

# Counter for files processed
COUNT=0

# Find all files matching PSI pattern in all subdirectories
while IFS= read -r -d '' file; do
    # Extract just the filename without path
    filename=$(basename "$file")
    
    # Extract PSI number using regex
    # This matches PSI followed by digits
    if [[ "$filename" =~ ^PSI([0-9]+)_ ]]; then
        psi_number="${BASH_REMATCH[1]}"
        psi_prefix="PSI${psi_number}"
        
        # Create subdirectory for this PSI number if it doesn't exist
        target_dir="$DEST_DIR/$psi_prefix"
        if [ ! -d "$target_dir" ]; then
            echo "Creating directory: $target_dir"
            mkdir -p "$target_dir"
        fi
        
        # Copy file to appropriate subdirectory
        echo "Moving: $filename -> $psi_prefix/"
        mv "$file" "$target_dir/"
        ((COUNT++))
    else
        echo "Skipping (no PSI pattern): $filename"
    fi
done < <(find "$SOURCE_DIR" -type f -name "PSI*" -print0)

echo "---"
echo "Total files organized: $COUNT"
echo "Created subdirectories in: $DEST_DIR"