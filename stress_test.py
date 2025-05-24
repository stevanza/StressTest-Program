#!/usr/bin/env python3
"""
Run Fixed Stress Test
====================

This script runs the fixed stress test with both thread and process mode support.
"""

import os
import sys
import shutil
import time

def main():
    print("="*80)
    print("FIXED STRESS TEST RUNNER")
    print("="*80)
    
    # Step 1: Backup original files
    print("\n1. Backing up original files...")
    
    try:
        # Backup file_server_pools.py
        if os.path.exists('file_server_pools.py'):
            shutil.copy2('file_server_pools.py', 'file_server_pools_backup.py')
            print("   ✓ Backed up file_server_pools.py")
        
        # Backup stress_test_runner.py if it exists
        if os.path.exists('stress_test_runner.py'):
            shutil.copy2('stress_test_runner.py', 'stress_test_runner_backup.py')
            print("   ✓ Backed up stress_test_runner.py")
    except Exception as e:
        print(f"   ⚠️  Warning: Could not backup files: {e}")
    
    # Step 2: Check if fixed files exist
    print("\n2. Checking for fixed files...")
    
    fixed_server_exists = os.path.exists('fixed_file_server_pools.py')
    fixed_runner_exists = os.path.exists('fixed_stress_test_runner.py')
    
    if not fixed_server_exists or not fixed_runner_exists:
        print("   ❌ Fixed files not found!")
        print("   Please ensure you have:")
        print("   - fixed_file_server_pools.py")
        print("   - fixed_stress_test_runner.py")
        print("\n   Copy the fixed code from the artifacts into these files.")
        return 1
    
    print("   ✓ Fixed files found")
    
    # Step 3: Replace original files with fixed versions
    print("\n3. Installing fixed files...")
    
    try:
        shutil.copy2('fixed_file_server_pools.py', 'file_server_pools.py')
        print("   ✓ Installed fixed file_server_pools.py")
        
        shutil.copy2('fixed_stress_test_runner.py', 'stress_test_runner.py')
        print("   ✓ Installed fixed stress_test_runner.py")
    except Exception as e:
        print(f"   ❌ Error installing fixed files: {e}")
        return 1
    
    # Step 4: Create necessary directories
    print("\n4. Creating necessary directories...")
    
    dirs_to_create = ['files', 'test_files']
    for dir_name in dirs_to_create:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"   ✓ Created {dir_name}/")
        else:
            print(f"   ✓ {dir_name}/ already exists")
    
    # Step 5: Run the stress test
    print("\n5. Starting stress test...")
    print("="*80)
    print("This will test both THREAD and PROCESS modes")
    print("Total configurations: 96 (48 thread + 48 process)")
    print("Estimated time: 30-60 minutes")
    print("="*80)
    
    response = input("\nDo you want to continue? (y/n): ").strip().lower()
    if response != 'y':
        print("Test cancelled.")
        return 0
    
    print("\nStarting stress test...\n")
    
    # Run the stress test
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'stress_test_runner.py'], check=False)
        
        if result.returncode == 0:
            print("\n✅ Stress test completed successfully!")
            print("\nCheck the following files for results:")
            print("  - fixed_stress_test_results.csv")
            print("  - fixed_stress_test_summary.csv")
            print("  - fixed_stress_test_report.txt")
        else:
            print(f"\n⚠️  Stress test exited with code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error running stress test: {e}")
        return 1
    
    # Step 6: Restore original files (optional)
    print("\n6. Cleanup options...")
    response = input("Do you want to restore the original files? (y/n): ").strip().lower()
    
    if response == 'y':
        try:
            if os.path.exists('file_server_pools_backup.py'):
                shutil.copy2('file_server_pools_backup.py', 'file_server_pools.py')
                print("   ✓ Restored original file_server_pools.py")
            
            if os.path.exists('stress_test_runner_backup.py'):
                shutil.copy2('stress_test_runner_backup.py', 'stress_test_runner.py')
                print("   ✓ Restored original stress_test_runner.py")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not restore files: {e}")
    
    print("\n✅ Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())