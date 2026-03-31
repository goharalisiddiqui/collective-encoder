#!/bin/bash
# Quick Start Script for Alanine Dipeptide Tutorial
# This script runs the complete workflow: training + analysis

set -e  # Exit on any error

echo "🧬 Alanine Dipeptide VAE Tutorial - Quick Start"
echo "==============================================="

# Check if we're in the right directory
if [[ ! -f "ala2_config.yaml" ]]; then
    echo "❌ Error: Please run this script from the ala2_tutorial directory"
    echo "💡 cd examples/ala2_tutorial/"
    exit 1
fi

# Check if data files exist
if [[ ! -f "data/ala2.tpr" ]] || [[ ! -f "data/ala2.xtc" ]]; then
    echo "❌ Error: Data files not found in data/ directory"
    echo "💡 Make sure ala2.tpr and ala2.xtc are in the data/ directory"
    exit 1
fi

# Check if collective-encoder is installed
if ! command -v collective-encoder-train &> /dev/null; then
    echo "❌ Error: collective-encoder not found in PATH"
    echo "💡 Install the package first:"
    echo "   cd ../../  # Go to project root"
    echo "   pip install -e ."
    echo "   cd examples/ala2_tutorial/"
    exit 1
fi

echo ""
echo "📋 Workflow:"
echo "  1. Quick training (debug mode) - ~2 minutes"
echo "  2. Full training - ~10-20 minutes"
echo "  3. Results analysis and visualization"
echo ""

# Offer choice
read -p "Run quick test (debug mode) or full training? [q/f/both]: " choice

case $choice in
    q|Q|quick|debug)
        echo ""
        echo "🚀 Running quick test (debug mode)..."
        collective-encoder-train --config ala2_config.yaml --debug
        echo ""
        echo "🎉 Debug completed!"
        echo ""
        echo "📁 Check results/debug_0 for results"
        exit 0
        ;;
    f|F|full)
        echo ""
        echo "⏳ Running full training..."
        collective-encoder-train --config ala2_config.yaml
        echo ""
        ;;
    b|B|both)
        echo ""
        echo "🚀 Step 1: Quick test (debug mode)..."
        collective-encoder-train --config ala2_config.yaml --debug
        echo ""
        echo "⏳ Step 2: Full training..."
        collective-encoder-train --config ala2_config.yaml
        echo ""
        ;;
    *)
        echo "❌ Invalid choice. Please run manually:"
        echo "   collective-encoder-train --config ala2_config.yaml --debug  # Quick test"
        echo "   collective-encoder-train --config ala2_config.yaml          # Full training"
        exit 1
        ;;
esac

echo ""
echo "🎉 Workflow completed!"
echo ""
echo "📁 Check these directories for results:"
echo "   📊 results/debug_0/                # Debug outputs"
echo "   📊 results/ala2_vae_1/             # Training outputs"
echo ""
