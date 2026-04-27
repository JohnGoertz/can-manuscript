#!/bin/bash

# Exit on error
set -e

echo "Generating predictions for all panels and axes..."

# Panel 0: Heatmap predictions
echo "Generating heatmap predictions..."
for axis in {0..5}; do
    echo "  Axis $axis"
    python predict_main.py 0 $axis
done

# Panel 1: Signal vs SNV predictions
echo "Generating signal vs SNV predictions..."
for axis in {0..5}; do
    echo "  Axis $axis"
    # Generate predictions for each WT copy level
    for wt_copies in {1..8}; do
        echo "    WT copies: $wt_copies"
        python predict_main.py 1 $axis --wt-copies $wt_copies
    done
done

# Panel 2: VAF predictions
echo "Generating VAF predictions..."
for axis in {0..5}; do
    echo "  Axis $axis"
    python predict_main.py 2 $axis
done

echo "All predictions generated successfully!" 