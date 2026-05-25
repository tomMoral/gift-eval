#!/usr/bin/env python
"""
Quick-start verification script for the Time Series Classification Benchmark.

This script verifies that all components are properly installed and configured.
Run this before running the full benchopt benchmark.

Usage:
    python benchmark_classification/verify_setup.py
"""

import sys
import importlib


def check_import(package_name, pip_name=None):
    """Check if a package is installed."""
    if pip_name is None:
        pip_name = package_name
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        print(f"✓ {pip_name:20} : {version}")
        return True
    except ImportError:
        print(f"✗ {pip_name:20} : NOT INSTALLED")
        return False


def main():
    """Verify all dependencies are installed."""
    print("\n" + "="*60)
    print("Time Series Classification Benchmark - Setup Verification")
    print("="*60 + "\n")
    
    print("Checking core dependencies...\n")
    
    checks = [
        ("benchopt", "benchopt"),
        ("sklearn", "scikit-learn"),
        ("numpy", "numpy"),
        ("torch", "PyTorch"),
        ("transformers", "transformers"),
        ("tslearn", "tslearn"),
    ]
    
    results = []
    for package, pip_name in checks:
        results.append(check_import(package, pip_name))
    
    print("\n" + "-"*60)
    
    if all(results):
        print("\n✓ All dependencies are installed!")
        print("\nYou can now run the benchmark:")
        print("  benchopt run benchmark_classification/")
        print("\nOr with a specific solver:")
        print("  benchopt run benchmark_classification/ -s MantisV2-RandomForest")
        return 0
    else:
        print("\n✗ Some dependencies are missing!")
        print("\nInstall missing dependencies with:")
        print("  pip install -e . && pip install tslearn transformers torch benchopt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
