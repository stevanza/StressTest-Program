#!/usr/bin/env python3
"""
File Server Stress Test Suite
============================

Complete setup and execution script for file server stress testing.

Requirements:
- Python 3.7+
- pandas
- matplotlib
- seaborn

Installation:
    pip install pandas matplotlib seaborn

Usage:
    python setup_and_run.py install-deps    # Install dependencies
    python setup_and_run.py quick-test      # Run quick test (5 combinations)
    python setup_and_run.py full-test       # Run full stress test (54 combinations)
    python setup_and_run.py analyze         # Analyze existing results
"""

import subprocess
import sys
import os
import time
import shutil
from pathlib import Path

def install_dependencies():
    """Install required dependencies"""
    dependencies = [
        'pandas>=1.3.0',
        'matplotlib>=3.3.0', 
        'seaborn>=0.11.0'
    ]
    
    print("Installing dependencies...")
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"✓ Installed {dep}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {dep}: {e}")
            return False
    
    print("All dependencies installed successfully!")
    return True

def create_project_structure():
    """Create necessary directories and files"""
    dirs = ['files', 'test_files', 'results', 'logs']
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_name}")

def run_quick_test():
    """Run a quick stress test with limited combinations"""
    print("=" * 50)
    print("RUNNING QUICK STRESS TEST")
    print("=" * 50)
    print("This will run a subset of tests to verify the system works.")
    print("Expected duration: 5-10 minutes")
    print()
    
    # Create a simplified test runner
    quick_test_script = '''
import time
import logging
from enhanced_client import FileClient, create_test_file, run_threading_stress_test
import pandas as pd

logging.basicConfig(level=logging.INFO)

def quick_stress_test():
    """Run quick stress test"""
    results = []
    
    # Create small test file
    print("Creating test file...")
    test_file = create_test_file("quick_test.bin", 1)  # 1MB file
    with open(test_file, 'rb') as f:
        file_data = f.read()
    
    # Test configurations (simplified)
    configs = [
        {'op': 'upload', 'workers': 1, 'server_port': 45000},
        {'op': 'upload', 'workers': 5, 'server_port': 45000},
        {'op': 'download', 'workers': 1, 'server_port': 45000},
        {'op': 'download', 'workers': 5, 'server_port': 45000},
    ]
    
    print(f"Running {len(configs)} test configurations...")
    
    for i, config in enumerate(configs, 1):
        print(f"\\nTest {i}/{len(configs)}: {config['op']} with {config['workers']} workers")
        
        try:
            # For download tests, we'd need to upload first
            if config['op'] == 'download':
                client = FileClient('localhost', config['server_port'])
                client.upload_file("quick_test.bin", file_data)
            
            result = run_threading_stress_test(
                config['op'],
                'localhost', 
                config['server_port'],
                "quick_test.bin",
                file_data,
                config['workers'],
                timeout=30
            )
            
            row = {
                'Test': i,
                'Operation': config['op'],
                'Workers': config['workers'],
                'Success': result['successful_workers'],
                'Failed': result['failed_workers'],
                'Avg_Time': result['avg_time_per_client'],
                'Throughput': result['throughput']
            }
            results.append(row)
            
            print(f"  Result: {result['successful_workers']} success, {result['failed_workers']} failed")
            print(f"  Throughput: {result['throughput']:.2f} B/s")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'Test': i,
                'Operation': config['op'], 
                'Workers': config['workers'],
                'Success': 0,
                'Failed': config['workers'],
                'Avg_Time': 0,
                'Throughput': 0
            })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('quick_test_results.csv', index=False)
    
    print("\\n" + "=" * 40)
    print("QUICK TEST RESULTS:")
    print("=" * 40)
    print(df.to_string(index=False))
    print(f"\\nResults saved to quick_test_results.csv")
    
    return len([r for r in results if r['Failed'] == 0])

if __name__ == "__main__":
    successful_tests = quick_stress_test()
    print(f"\\nQuick test completed: {successful_tests} successful configurations")
'''
    
    with open('quick_test_runner.py', 'w') as f:
        f.write(quick_test_script)
    
    # Note: This is a simplified version that doesn't start servers
    # In practice, you'd need to start the server first
    print("Quick test script created: quick_test_runner.py")
    print("Note: You need to start a server first with:")
    print("  python file_server_pools.py thread 5")
    print("Then run: python quick_test_runner.py")

def run_full_test():
    """Run complete stress test suite"""
    print("=" * 50)
    print("RUNNING FULL STRESS TEST SUITE")
    print("=" * 50)
    print("This will run all 54 test combinations.")
    print("Expected duration: 2-4 hours")
    print("Results will be saved in the 'results' directory")
    print()
    
    confirm = input("Are you sure you want to proceed? (y/N): ")
    if confirm.lower() != 'y':
        print("Test cancelled.")
        return
    
    try:
        from stress_test_runner import StressTestRunner
        runner = StressTestRunner()
        runner.run_all_tests()
    except ImportError:
        print("Error: stress_test_runner module not found.")
        print("Make sure all files are in the same directory.")
    except Exception as e:
        print(f"Error running full test: {e}")

def analyze_results():
    """Analyze existing test results"""
    import glob
    
    # Find CSV files
    csv_files = glob.glob("*results*.csv")
    
    if not csv_files:
        print("No results files found.")
        print("Looking for files matching pattern: *results*.csv")
        return
    
    print("Found results files:")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {file}")
    
    if len(csv_files) == 1:
        selected_file = csv_files[0]
    else:
        try:
            choice = int(input(f"Select file to analyze (1-{len(csv_files)}): ")) - 1
            selected_file = csv_files[choice]
        except (ValueError, IndexError):
            print("Invalid selection.")
            return
    
    print(f"Analyzing {selected_file}...")
    
    try:
        from server_manager import ResultsAnalyzer
        analyzer = ResultsAnalyzer(selected_file)
        analyzer.find_optimal_configurations()
        analyzer.export_summary_table()
        print("Analysis complete! Check the generated files.")
    except ImportError:
        print("Error: server_manager module not found.")
    except Exception as e:
        print(f"Error analyzing results: {e}")

def show_help():
    """Show help information"""
    print(__doc__)

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1]
    
    # Ensure project structure exists
    create_project_structure()
    
    if command == "install-deps":
        install_dependencies()
    elif command == "quick-test":
        run_quick_test()
    elif command == "full-test":
        run_full_test()
    elif command == "analyze":
        analyze_results()
    elif command == "help":
        show_help()
    else:
        print(f"Unknown command: {command}")
        show_help()

if __name__ == "__main__":
    main()