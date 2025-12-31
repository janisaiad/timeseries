#!/bin/bash
# Script to run all analysis scripts and generate plots


echo "Running jump_density_poland.py..."
uv run python jump_density_poland.py

echo ""
echo "Running reproduce_poland.py..."
uv run python reproduce_poland.py

echo ""
echo "Running poland_comparison_scattering.py..."
uv run python poland_comparison_scattering.py

echo ""
echo "Running reproduce_multi_stock.py..."
uv run python reproduce_multi_stock.py

echo ""
echo "Running reproduce_hong_kong.py..."
uv run python reproduce_hong_kong.py

echo ""
echo "Running jump_density_hongkong.py..."
uv run python jump_density_hongkong.py

echo ""
echo "All scripts completed!"

