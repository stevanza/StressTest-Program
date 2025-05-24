#!/usr/bin/env python3
"""
Modified File Server Stress Test Suite
====================================

Complete setup and execution script for file server stress testing
with 108 combinations (2×2×3×3×3).

Requirements:
- Python 3.7+
- pandas
- matplotlib
- seaborn

Installation:
    pip install pandas matplotlib seaborn

Usage:
    python setup_and_run.py install-deps    # Install dependencies
    python setup_and_run.py quick-test      # Run quick test (4 combinations)
    python setup_and_run.py full-test       # Run full stress test (108 combinations)
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
    """Run a quick stress test with 4 combinations"""
    print("=" * 50)
    print("RUNNING QUICK STRESS TEST")
    print("=" * 50)
    print("This will run 4 test combinations to verify the system works.")
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
    """Run quick stress test with 4 combinations"""
    results = []
    
    # Create small test file
    print("Creating test file...")
    test_file = create_test_file("quick_test.bin", 10)  # 10MB file
    with open(test_file, 'rb') as f:
        file_data = f.read()
    
    # Test configurations (4 combinations: 2 operations × 2 worker configs)
    configs = [
        {'op': 'upload', 'workers': 1, 'server_port': 45000, 'mode': 'thread'},
        {'op': 'upload', 'workers': 5, 'server_port': 45000, 'mode': 'thread'},
        {'op': 'download', 'workers': 1, 'server_port': 45000, 'mode': 'thread'},
        {'op': 'download', 'workers': 5, 'server_port': 45000, 'mode': 'thread'},
    ]
    
    print(f"Running {len(configs)} test configurations...")
    
    for i, config in enumerate(configs, 1):
        print(f"\\nTest {i}/{len(configs)}: {config['op']} with {config['workers']} workers ({config['mode']} mode)")
        
        try:
            # For download tests, upload first
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
                'Nomor': i,
                'Execution_Mode': config['mode'],
                'Operasi': config['op'],
                'Volume': 10,  # 10MB test file
                'Jumlah_Client_Worker_Pool': config['workers'],
                'Jumlah_Server_Worker_Pool': 3,  # Default server workers
                'Waktu_Total_Per_Client': result['avg_time_per_client'],
                'Throughput_Per_Client': result['throughput'],
                'Client_Worker_Sukses': result['successful_workers'],
                'Client_Worker_Gagal': result['failed_workers'],
                'Throughput_MB_per_sec': result['throughput'] / (1024 * 1024)
            }
            results.append(row)
            
            print(f"  Result: {result['successful_workers']} success, {result['failed_workers']} failed")
            print(f"  Throughput: {result['throughput'] / (1024 * 1024):.2f} MB/s")
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'Nomor': i,
                'Execution_Mode': config['mode'],
                'Operasi': config['op'], 
                'Volume': 10,
                'Jumlah_Client_Worker_Pool': config['workers'],
                'Jumlah_Server_Worker_Pool': 3,
                'Waktu_Total_Per_Client': 0,
                'Throughput_Per_Client': 0,
                'Client_Worker_Sukses': 0,
                'Client_Worker_Gagal': config['workers'],
                'Throughput_MB_per_sec': 0
            })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('quick_test_results.csv', index=False)
    
    print("\\n" + "=" * 50)
    print("QUICK TEST RESULTS:")
    print("=" * 50)
    print(df.to_string(index=False))
    print(f"\\nResults saved to quick_test_results.csv")
    
    return len([r for r in results if r['Client_Worker_Gagal'] == 0])

if __name__ == "__main__":
    successful_tests = quick_stress_test()
    print(f"\\nQuick test completed: {successful_tests} successful configurations")
'''
    
    with open('quick_test_runner.py', 'w') as f:
        f.write(quick_test_script)
    
    print("Quick test script created: quick_test_runner.py")
    print("\nTo run the quick test:")
    print("1. Start a server first: python file_server_pools.py thread 5")
    print("2. In another terminal, run: python quick_test_runner.py")

def run_full_test():
    """Run complete stress test suite with 108 combinations"""
    print("=" * 60)
    print("RUNNING FULL STRESS TEST SUITE - 108 COMBINATIONS")
    print("=" * 60)
    print("This will run all 108 test combinations:")
    print("- Execution modes: thread, process (2)")
    print("- Operations: download, upload (2)")
    print("- File volumes: 10MB, 50MB, 100MB (3)")
    print("- Client workers: 1, 5, 50 (3)")
    print("- Server workers: 1, 5, 50 (3)")
    print("Total: 2×2×3×3×3 = 108 combinations")
    print()
    print("Expected duration: 3-6 hours")
    print("Results will be saved as single CSV file")
    print()
    
    confirm = input("Are you sure you want to proceed? (y/N): ")
    if confirm.lower() != 'y':
        print("Test cancelled.")
        return
    
    try:
        # Check if modified stress test runner exists
        if not os.path.exists('stress_test_runner.py'):
            print("Error: stress_test_runner.py not found.")
            print("Please ensure the modified stress test runner is in the current directory.")
            return
            
        # Import and run the modified stress test runner
        print("Starting 108-combination stress test...")
        print("This will take several hours. Progress will be logged.")
        print("You can monitor progress in the log output.")
        print()
        
        # Run the modified stress test runner
        from stress_test_runner import ModifiedStressTestRunner
        runner = ModifiedStressTestRunner()
        runner.run_all_tests()
        
        print("\n" + "=" * 60)
        print("✅ FULL STRESS TEST COMPLETED!")
        print("📊 Check these files for results:")
        print("  - stress_test_results_108_combinations.csv (complete results)")
        print("  - stress_test_summary_108.csv (summary)")
        print("  - stress_test_final_report.txt (detailed report)")
        print("=" * 60)
        
    except ImportError:
        print("Error: Modified stress test runner not found.")
        print("Make sure all required files are in the same directory.")
    except Exception as e:
        print(f"Error running full test: {e}")

def analyze_results():
    """Analyze existing test results"""
    import glob
    
    # Look for CSV files with results
    csv_files = glob.glob("*results*.csv") + glob.glob("*stress_test*.csv")
    
    if not csv_files:
        print("No results files found.")
        print("Looking for files matching patterns: *results*.csv, *stress_test*.csv")
        return
    
    print("Found results files:")
    for i, file in enumerate(csv_files, 1):
        size = os.path.getsize(file) / 1024  # KB
        print(f"  {i}. {file} ({size:.1f} KB)")
    
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
        import pandas as pd
        df = pd.read_csv(selected_file)
        
        print("\n" + "=" * 50)
        print("ANALYSIS RESULTS")
        print("=" * 50)
        
        # Basic statistics
        print(f"Total tests: {len(df)}")
        if 'Client_Worker_Gagal' in df.columns:
            successful = len(df[df['Client_Worker_Gagal'] == 0])
            print(f"Successful tests: {successful}")
            print(f"Failed tests: {len(df) - successful}")
        elif 'Client_Failed' in df.columns:  # Backward compatibility
            successful = len(df[df['Client_Failed'] == 0])
            print(f"Successful tests: {successful}")
            print(f"Failed tests: {len(df) - successful}")
        
        # Performance statistics - check for different column name variations
        if 'Throughput_MB_Per_Second' in df.columns:
            throughput_col = 'Throughput_MB_Per_Second'
        elif 'Throughput_Per_Client' in df.columns:
            # Convert to MB/s if it's in bytes/s
            if df['Throughput_Per_Client'].max() > 1000000:  # Likely bytes/s
                df['Throughput_MB_Per_Second'] = df['Throughput_Per_Client'] / (1024 * 1024)
            else:  # Already in MB/s
                df['Throughput_MB_Per_Second'] = df['Throughput_Per_Client']
            throughput_col = 'Throughput_MB_Per_Second'
        elif 'Throughput_Bytes_Per_Second' in df.columns:
            throughput_col = 'Throughput_Bytes_Per_Second'
            df['Throughput_MB_Per_Second'] = df[throughput_col] / (1024 * 1024)
            throughput_col = 'Throughput_MB_Per_Second'
        else:
            print("No throughput data found in results.")
            return
            
        print(f"\\nThroughput Statistics (MB/s):")
        print(f"  Average: {df[throughput_col].mean():.2f}")
        print(f"  Maximum: {df[throughput_col].max():.2f}")
        print(f"  Minimum: {df[throughput_col].min():.2f}")
        
        # Top performers
        print(f"\\nTop 5 Best Performing Configurations:")
        top5 = df.nlargest(5, throughput_col)
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            config_info = []
            # Check for new column names first, then fall back to old ones
            col_mapping = {
                'Execution_Mode': 'Execution_Mode',
                'Operasi': 'Operation', 
                'Volume': 'Volume_MB',
                'Jumlah_Client_Worker_Pool': 'Client_Workers',
                'Jumlah_Server_Worker_Pool': 'Server_Workers'
            }
            
            for new_col, old_col in col_mapping.items():
                if new_col in row:
                    config_info.append(f"{new_col.replace('_', ' ')}: {row[new_col]}")
                elif old_col in row:
                    config_info.append(f"{old_col.replace('_', ' ')}: {row[old_col]}")
            
            print(f"  {i}. {', '.join(config_info)}")
            print(f"     Throughput: {row[throughput_col]:.2f} MB/s")
        
        # Generate advanced analysis if server_manager is available
        try:
            from server_manager import ResultsAnalyzer
            analyzer = ResultsAnalyzer(selected_file)
            
            print("\\nGenerating detailed analysis plots and reports...")
            analyzer.find_optimal_configurations()
            analyzer.export_summary_table()
            print("✓ Advanced analysis completed!")
            
        except ImportError:
            print("\\nNote: For advanced analysis with plots, ensure server_manager.py is available.")
        except Exception as e:
            print(f"Advanced analysis failed: {e}")
            
    except Exception as e:
        print(f"Error analyzing results: {e}")

def show_system_info():
    """Show system information and requirements"""
    print("=" * 60)
    print("FILE SERVER STRESS TEST SYSTEM INFO")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    print()
    
    # Check for required files
    required_files = [
        'file_server_pools.py',
        'enhanced_client.py', 
        'stress_test_runner.py',
        'server_manager.py'
    ]
    
    print("Required files status:")
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"  ✓ {file} ({size:.1f} KB)")
        else:
            print(f"  ✗ {file} (missing)")
    
    print()
    
    # Check for required packages
    packages = ['pandas', 'matplotlib', 'seaborn']
    print("Required packages status:")
    for package in packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (not installed)")

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
    elif command == "info":
        show_system_info()
    elif command == "help":
        show_help()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: install-deps, quick-test, full-test, analyze, info, help")

if __name__ == "__main__":
    main()