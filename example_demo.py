#!/usr/bin/env python3
"""
Example Demo Script
==================

This script demonstrates the complete file server stress testing workflow
with a simple example that you can run to verify everything works.

Usage: python example_demo.py
"""

import os
import sys
import time
import threading
import subprocess
import json
from pathlib import Path

def create_demo_environment():
    """Create demo environment with necessary directories"""
    print("🔧 Setting up demo environment...")
    
    # Create directories
    dirs = ['files', 'test_files', 'demo_results']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   ✓ Created {dir_name}/")
    
    print("   ✓ Demo environment ready!")

def create_small_test_file():
    """Create a small test file for demo"""
    print("📄 Creating demo test file...")
    
    test_file = Path('test_files/demo_file.txt')
    content = "Hello World! This is a demo file for testing.\n" * 100  # ~4KB file
    
    with open(test_file, 'w') as f:
        f.write(content)
    
    size = test_file.stat().st_size
    print(f"   ✓ Created demo_file.txt ({size} bytes)")
    return str(test_file)

def start_demo_server():
    """Start a simple demo server"""
    print("🚀 Starting demo server...")
    
    # Create simple server script
    server_code = '''
import sys
import os
sys.path.append('.')

from file_server_pools import ThreadPoolServer
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    server = ThreadPoolServer(server_port=45000, max_workers=3)
    print("Demo server starting on port 45000...")
    try:
        server.start()
    except KeyboardInterrupt:
        print("Demo server stopped.")
        server.stop()
'''
    
    with open('demo_server.py', 'w') as f:
        f.write(server_code)
    
    # Start server in background
    process = subprocess.Popen([sys.executable, 'demo_server.py'])
    time.sleep(3)  # Wait for server to start
    
    print("   ✓ Demo server started on port 45000")
    return process

def run_demo_client_tests():
    """Run demo client tests"""
    print("🧪 Running demo client tests...")
    
    # Import after ensuring modules are available
    try:
        from enhanced_client import FileClient
    except ImportError:
        print("   ❌ Error: enhanced_client module not found")
        return False
    
    client = FileClient('localhost', 45000, timeout=10)
    
    # Test 1: List files
    print("   📋 Testing LIST operation...")
    success, files, time_taken = client.list_files()
    if success:
        print(f"      ✓ LIST successful ({time_taken:.2f}s): {files}")
    else:
        print(f"      ❌ LIST failed: {files}")
    
    # Test 2: Upload file
    print("   📤 Testing UPLOAD operation...")
    test_file = 'test_files/demo_file.txt'
    success, message, time_taken, bytes_transferred = client.upload_file(test_file)
    if success:
        print(f"      ✓ UPLOAD successful ({bytes_transferred} bytes in {time_taken:.2f}s)")
    else:
        print(f"      ❌ UPLOAD failed: {message}")
    
    # Test 3: Download file  
    print("   📥 Testing DOWNLOAD operation...")
    success, message, time_taken, bytes_transferred = client.download_file(
        'demo_file.txt', 'demo_results/downloaded_demo_file.txt'
    )
    if success:
        print(f"      ✓ DOWNLOAD successful ({bytes_transferred} bytes in {time_taken:.2f}s)")
    else:
        print(f"      ❌ DOWNLOAD failed: {message}")
    
    # Test 4: List files again
    print("   📋 Testing LIST operation (after upload)...")
    success, files, time_taken = client.list_files()
    if success:
        print(f"      ✓ LIST successful ({time_taken:.2f}s): {files}")
    else:
        print(f"      ❌ LIST failed: {files}")
    
    return True

def run_mini_stress_test():
    """Run a mini stress test"""
    print("⚡ Running mini stress test...")
    
    try:
        from enhanced_client import run_threading_stress_test, create_test_file
    except ImportError:
        print("   ❌ Error: Required modules not found")
        return False
    
    # Create tiny test file
    test_file = create_test_file("mini_stress_test.bin", 1)  # 1MB
    with open(test_file, 'rb') as f:
        file_data = f.read()
    
    print("   🔄 Running concurrent upload test (3 workers)...")
    result = run_threading_stress_test(
        operation='upload',
        host='localhost',
        port=45000, 
        filename='mini_stress_test.bin',
        file_data=file_data,
        num_workers=3,
        timeout=30
    )
    
    print(f"      ✓ Completed: {result['successful_workers']} success, {result['failed_workers']} failed")
    print(f"      📊 Throughput: {result['throughput']:.2f} B/s")
    print(f"      ⏱️  Avg time: {result['avg_time_per_client']:.2f}s")
    
    return True

def cleanup_demo():
    """Clean up demo artifacts"""
    print("🧹 Cleaning up demo files...")
    
    files_to_remove = [
        'demo_server.py',
        'test_files/demo_file.txt',
        'test_files/mini_stress_test.bin',
        'demo_results/downloaded_demo_file.txt'
    ]
    
    for file_path in files_to_remove:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"   ✓ Removed {file_path}")
        except Exception as e:
            print(f"   ⚠️  Could not remove {file_path}: {e}")

def main():
    """Main demo function"""
    print("=" * 60)
    print("🎭 FILE SERVER STRESS TEST DEMO")
    print("=" * 60)
    print("This demo will:")
    print("1. Setup environment")
    print("2. Start a demo server")
    print("3. Run basic client tests")
    print("4. Run mini stress test")
    print("5. Clean up")
    print()
    
    server_process = None
    
    try:
        # Step 1: Setup
        create_demo_environment()
        print()
        
        # Step 2: Create test file
        create_small_test_file()
        print()
        
        # Step 3: Start server
        server_process = start_demo_server()
        print()
        
        # Step 4: Run client tests
        if run_demo_client_tests():
            print("   🎉 Basic client tests completed!")
        print()
        
        # Step 5: Run mini stress test
        if run_mini_stress_test():
            print("   🎉 Mini stress test completed!")
        print()
        
        print("=" * 60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Run quick test: python setup_and_run.py quick-test")
        print("2. Run full test: python setup_and_run.py full-test")
        print("3. Analyze results: python setup_and_run.py analyze")
        print()
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all required files are in the same directory")
        print("2. Check if port 45000 is available")
        print("3. Install dependencies: python setup_and_run.py install-deps")
    finally:
        # Clean up server
        if server_process:
            print("🛑 Stopping demo server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait()
            print("   ✓ Demo server stopped")
        
        # Clean up files
        cleanup_demo()
        print("   ✓ Cleanup completed")

if __name__ == "__main__":
    main()