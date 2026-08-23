#!/bin/bash

set -e

VENV_DIR="venv"

# uv installs to ~/.local/bin -- no sudo needed
if ! command -v uv &> /dev/null; then
    echo "Installing uv (no sudo needed)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing Python 3.11 via uv (portable build in \$HOME, no sudo needed)..."
uv python install 3.11

echo "Creating virtual environment with Python 3.11..."
uv venv --python 3.11 "$VENV_DIR"

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "Verifying plantcv imports correctly..."
python -c "from plantcv import plantcv as pcv; import scipy; print(f'plantcv OK, scipy {scipy.__version__}')"

echo
echo "Setup completed successfully!"
echo
echo "To activate the environment later, run:"
echo "source $VENV_DIR/bin/activate"


echo
echo "Setup completed successfully!"
echo
echo "To activate the environment later, run:"
echo "source $VENV_DIR/bin/activate"