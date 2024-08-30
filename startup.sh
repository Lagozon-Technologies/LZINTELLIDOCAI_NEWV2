#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Create a directory in /home for custom installations
mkdir -p /home/site/custom

# Change to the custom directory
cd /home/site/custom

# Export the PATH to include the custom directory
echo 'export PATH=$PATH:/home/site/custom' >> ~/.bashrc

# Function to check if a command is available
# command_exists() {
#     command -v "$1" >/dev/null 2>&1
# }

# Function to check if a package is installed
# package_installed() {
#     dpkg -s "$1" >/dev/null 2>&1
# }

# Function to check if a Python package is installed
python_package_installed() {
    pip show --user "$1" >/dev/null 2>&1
}

# Check if dpkg is available, if not, try to install it
# if ! command_exists dpkg; then
#     echo "dpkg is not installed. Attempting to install it..."
#     if command_exists apt-get; then
#         apt-get update && apt-get install -y dpkg
#     elif command_exists yum; then
#         yum install -y dpkg
#     else
#         echo "Unable to install dpkg. No supported package manager found."
#         exit 1
#     fi
# fi

# Update package lists
apt-get update

# List of required system packages
packages=(
    "libmagic-dev"
	"poppler-utils"
	"tesseract-ocr"
	"libreoffice"
)

# Install system packages if not already installed
for package in "${packages[@]}"; do
    if ! package_installed "$package"; then
        echo "Installing $package..."
        apt-get install -y "$package"
    else
        echo "$package is already installed."
    fi
done

# Set up a persistent location for Python packages
# export PYTHONUSERBASE=/home/site/pythonpackages

# Ensure the directory exists
# mkdir -p $PYTHONUSERBASE

# Determine the Python version
# PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

# Install Python packages if not already installed
# python_packages=(
#     "dlib"
#     "face-recognition"
# )

# for package in "${python_packages[@]}"; do
#     if ! python_package_installed "$package"; then
#         echo "Installing $package..."
#         pip install --user "$package"
#     else
#         echo "$package is already installed."
#     fi
# done

# Update PATH and PYTHONPATH to include the persistent package location
# export PATH=$PYTHONUSERBASE/bin:$PATH
# export PYTHONPATH=$PYTHONUSERBASE/lib/python$PYTHON_VERSION/site-packages:$PYTHONPATH

# Start the Flask app
streamlit run new_main.py --server.port 8000 --server.address 0.0.0.0
