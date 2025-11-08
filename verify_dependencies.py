#!/usr/bin/env python3
"""
Dependency Verification Script for Eudia Legal AI System

This script checks if all required dependencies are installed
and provides a detailed report.
"""

import sys
import importlib
from typing import Dict, List, Tuple

REQUIRED_PACKAGES = {
    # PDF Processing
    'pdfplumber': '0.11.0',
    
    # Machine Learning & Transformers
    'torch': '2.0.0',
    'transformers': '4.30.0',
    'sentence_transformers': '5.0.0',
    
    # Vector Search
    'faiss': '1.8.0',  # faiss-cpu or faiss-gpu
    
    # Scientific Computing
    'numpy': '1.24.0',
    'scipy': '1.10.0',
    'sklearn': '1.3.0',  # scikit-learn
    'pandas': '2.0.0',
    
    # LangChain
    'langchain': '0.3.0',
    'langchain_community': '0.4.0',
    'langchain_core': '1.0.0',
    
    # Graph & Visualization
    'networkx': '3.0',
    'matplotlib': '3.7.0',
    
    # Utilities
    'requests': '2.28.0',
}

OPTIONAL_PACKAGES = {
    'pytest': '7.4.0',
    'PIL': '9.0.0',  # Pillow
}


def check_package(package_name: str, min_version: str) -> Tuple[bool, str, str]:
    """
    Check if a package is installed and meets minimum version requirement.
    
    Returns:
        (is_installed, installed_version, status_message)
    """
    try:
        module = importlib.import_module(package_name)
        
        # Get version
        version = None
        for attr in ['__version__', 'VERSION', 'version']:
            if hasattr(module, attr):
                version = getattr(module, attr)
                if isinstance(version, tuple):
                    version = '.'.join(map(str, version))
                break
        
        if version is None:
            return True, 'unknown', f'✓ {package_name} is installed (version unknown)'
        
        # Simple version comparison (works for most cases)
        installed_parts = [int(x) for x in version.split('.')[:3] if x.isdigit()]
        required_parts = [int(x) for x in min_version.split('.')[:3] if x.isdigit()]
        
        if installed_parts >= required_parts:
            return True, version, f'✓ {package_name} {version} (>= {min_version})'
        else:
            return False, version, f'✗ {package_name} {version} (need >= {min_version})'
            
    except ImportError:
        return False, 'not installed', f'✗ {package_name} is NOT installed'
    except Exception as e:
        return False, 'error', f'? {package_name} check failed: {str(e)}'


def main():
    """Run dependency verification."""
    print("=" * 70)
    print("EUDIA LEGAL AI SYSTEM - DEPENDENCY VERIFICATION")
    print("=" * 70)
    print()
    
    # Check Python version
    print(f"Python Version: {sys.version}")
    print()
    
    # Check required packages
    print("REQUIRED PACKAGES:")
    print("-" * 70)
    
    all_installed = True
    missing_packages = []
    outdated_packages = []
    
    for package, min_version in REQUIRED_PACKAGES.items():
        is_installed, version, message = check_package(package, min_version)
        print(message)
        
        if not is_installed:
            all_installed = False
            if version == 'not installed':
                missing_packages.append(package)
            else:
                outdated_packages.append((package, version, min_version))
    
    print()
    
    # Check optional packages
    print("OPTIONAL PACKAGES:")
    print("-" * 70)
    
    for package, min_version in OPTIONAL_PACKAGES.items():
        _, _, message = check_package(package, min_version)
        print(message)
    
    print()
    print("=" * 70)
    
    # Summary
    if all_installed:
        print("✓ ALL REQUIRED DEPENDENCIES ARE INSTALLED!")
        print()
        print("You can now run the Eudia Legal AI system.")
        print()
        print("Quick start:")
        print("  python lexai/quick_start_inlegalbert.py")
        print("  pytest tests/")
        return 0
    else:
        print("✗ SOME REQUIRED DEPENDENCIES ARE MISSING OR OUTDATED")
        print()
        
        if missing_packages:
            print("Missing packages:")
            for pkg in missing_packages:
                # Map package import name to pip install name
                pip_name = pkg
                if pkg == 'sklearn':
                    pip_name = 'scikit-learn'
                elif pkg == 'PIL':
                    pip_name = 'Pillow'
                print(f"  pip install {pip_name}")
        
        if outdated_packages:
            print()
            print("Outdated packages:")
            for pkg, current, required in outdated_packages:
                pip_name = pkg
                if pkg == 'sklearn':
                    pip_name = 'scikit-learn'
                print(f"  pip install --upgrade {pip_name}>={required}  # current: {current}")
        
        print()
        print("Or install all requirements:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
