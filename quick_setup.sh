#!/bin/bash

echo "================================================================================"
echo "JARVIS INFINITE CAPACITY - QUICK SETUP"
echo "================================================================================"
echo ""
echo "This will install dependencies and run the comprehensive benchmark suite."
echo ""
read -p "Press Enter to continue..."

echo ""
echo "[1/3] Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "[2/3] Running comprehensive benchmark suite..."
echo "This will test:"
echo "  - Scaling to 1000+ adapters"
echo "  - Router accuracy"
echo "  - Conflict resolution"
echo "  - Transfer learning"
echo "  - Storage efficiency"
echo ""
python benchmark_suite.py

echo ""
echo "[3/3] Creating visualizations..."
python visualization.py

echo ""
echo "================================================================================"
echo "SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - benchmark_results.json (raw data)"
echo "  - BENCHMARK_REPORT.md (detailed report)"
echo ""
echo "To see the visual demo:"
echo "  python demo_ultimate_visual.py"
echo ""
