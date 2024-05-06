#!/usr/bin/env bash

# Stop on the first error
set -e

# Define directories relative to this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"
PYTHON_PACKAGE_DIR="$SCRIPT_DIR/src"


# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory at ${OUTPUT_DIR}"
    mkdir -p "$OUTPUT_DIR"
fi

# # Setup Python 3.9 (assuming it's already installed)
# echo "Setting up Python environment..."
# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt-get update
# sudo apt-get install -y python3.9 python3.9-dev python3.9-distutils

# # Ensure pip is installed and upgraded
# sudo apt-get install -y python3-pip
# python3.9 -m pip install --upgrade pip

# Install Python dependencies
# echo "Installing Python dependencies..."
# python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

# Run the Python script as a module
echo "Running Python script..."
cd $PYTHON_PACKAGE_DIR
python -m process

# Replace 'yourpackagename.yourscriptname' with the actual path to your script,
# e.g., 'my_package.my_script' if your directory structure looks like:
# ├───my_package
# │   ├───__init__.py
# │   └───my_script.py

echo "Script execution completed successfully. Check output in ${OUTPUT_DIR}"
