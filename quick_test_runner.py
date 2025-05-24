#!/usr/bin/env python3
"""
Improved Quick Test Runner for File Server Stress Testing
=========================================================

This script runs a simplified stress test with automatic server management,
better error handling, and comprehensive reporting.

Features:
- Automatic server startup and shutdown
- Port availability checking
- Comprehensive error handling
- Detailed progress reporting
- Automatic cleanup
- Fallback options when dependencies are missing

Usage:
    python improved_quick_test.py
"""

import os
import sys
import time
import subprocess
import socket
import threading
import logging
import signal
from pathlib import Path
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImprovedQuickTestRunner:
    def __init__(self):
        self.server_processes = {}
        self.test_files = {}
        self.results = []
        self.cleanup_files = []
        self.base_dir = Path.cwd()
        
    def check_dependencies(self):
        """Check if required dependencies and files are available"""
        required_modules = ['pandas']
        optional_modules = ['matplotlib', 'seaborn']
        required_files = ['file_server_pools.py']
        
        missing_modules = []
        missing_files = []
        
        # Check required modules
        for module in required_modules:
            try:
                __import__(module)
                logger.info(f"✓ Required module '{module}' found")
            except ImportError:
                missing_modules.append(module)
                logger.warning(f"✗ Required module '{module}' not found")
        
        # Check optional modules
        for module in optional_modules:
            try:
                __import__(module)
                logger.info(f"✓ Optional module '{module}' found")
            except ImportError:
                logger.info(f"- Optional module '{module}' not found (will skip advanced features)")
        
        # Check required files
        for file in required_files:
            if os.path.exists(file):
                logger.info(f"✓ Required file '{file}' found")
            else:
                missing_files.append(file)
                logger.error(f"✗ Required file '{file}' not found")
        
        if missing_modules or missing_files:
            logger.error("Missing dependencies detected!")
            if missing_modules:
                logger.error(f"Install missing modules: pip install {' '.join(missing_modules)}")
            if missing_files:
                logger.error(f"Missing files: {', '.join(missing_files)}")
            return False
        
        return True
    
    def find_available_port(self, start_port=45000, max_attempts=10):
        """Find an available port starting from start_port"""
        for i in range(max_attempts):
            port = start_port + i
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                    logger.info(f"Found available port: {port}")
                    return port
                except OSError:
                    continue
        
        raise RuntimeError(f"Could not find available port after {max_attempts} attempts")
    
    def create_test_file(self, filename, size_mb):
        """Create a test file of specified size"""
        filepath = self.base_dir / filename
        size_bytes = size_mb * 1024 * 1024
        
        logger.info(f"Creating test file: {filename} ({size_mb}MB)")
        
        try:
            with open(filepath, 'wb') as f:
                # Write in chunks to avoid memory issues
                chunk_size = 1024 * 1024  # 1MB chunks
                remaining = size_bytes
                
                while remaining > 0:
                    chunk = min(chunk_size, remaining)
                    f.write(os.urandom(chunk))
                    remaining -= chunk
            
            self.cleanup_files.append(filepath)
            logger.info(f"✓ Created {filename} ({size_bytes} bytes)")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to create test file {filename}: {e}")
            raise
    
    def start_server(self, server_type='thread', workers=3, port=None):
        """Start file server process"""
        if port is None:
            port = self.find_available_port()
        
        cmd = [sys.executable, "file_server_pools.py", server_type, str(workers)]
        
        logger.info(f"Starting {server_type} server on port {port} with {workers} workers")
        
        try:
            # Set environment variable for port if server supports it
            env = os.environ.copy()
            env['SERVER_PORT'] = str(port)
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            # Wait for server to start
            logger.info("Waiting for server to start...")
            time.sleep(5)
            
            # Check if server is running
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(f"Server failed to start: {stderr}")
                raise RuntimeError(f"Server process exited with code {process.returncode}")
            
            # Test server connectivity
            if self.test_server_connection('localhost', port):
                logger.info(f"✓ Server started successfully on port {port}")
                self.server_processes[port] = process
                return process, port
            else:
                logger.error("Server started but not responding to connections")
                process.terminate()
                raise RuntimeError("Server not responding")
                
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise
    
    def test_server_connection(self, host, port, timeout=10):
        """Test if server is accepting connections"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                result = s.connect_ex((host, port))
                return result == 0
        except Exception:
            return False
    
    def stop_server(self, process, port):
        """Stop server process"""
        if process and process.poll() is None:
            logger.info(f"Stopping server on port {port}")
            process.terminate()
            try:
                process.wait(timeout=5)
                logger.info("✓ Server stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Server did not stop gracefully, killing...")
                process.kill()
                process.wait()
        
        if port in self.server_processes:
            del self.server_processes[port]
    
    def simple_file_client(self, host, port, operation, filename, file_data=None):
        """Simple file client for testing (fallback if enhanced_client not available)"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(30)
                s.connect((host, port))
                
                if operation == 'upload':
                    if file_data is None:
                        raise ValueError("file_data required for upload")
                    
                    # Send upload request
                    request = f"UPLOAD {filename} {len(file_data)}\n"
                    s.send(request.encode())
                    
                    # Send file data
                    s.sendall(file_data)
                    
                    # Get response
                    response = s.recv(1024).decode().strip()
                    return response.startswith('OK')
                
                elif operation == 'download':
                    # Send download request
                    request = f"DOWNLOAD {filename}\n"
                    s.send(request.encode())
                    
                    # Get response header
                    response = s.recv(1024).decode().strip()
                    if not response.startswith('OK'):
                        return False
                    
                    # Parse file size from response
                    try:
                        size = int(response.split()[1])
                    except (IndexError, ValueError):
                        return False
                    
                    # Receive file data
                    received = 0
                    data = b''
                    while received < size:
                        chunk = s.recv(min(8192, size - received))
                        if not chunk:
                            break
                        data += chunk
                        received += len(chunk)
                    
                    return len(data) == size
                
        except Exception as e:
            logger.error(f"Client operation failed: {e}")
            return False
    
    def run_simple_stress_test(self, operation, host, port, filename, file_data, num_workers, timeout=60):
        """Run simple stress test using threading"""
        results = {
            'successful_workers': 0,
            'failed_workers': 0,
            'total_time': 0,
            'avg_time_per_client': 0,
            'throughput': 0
        }
        
        start_time = time.time()
        worker_results = []
        threads = []
        
        def worker_task(worker_id):
            worker_start = time.time()
            try:
                success = self.simple_file_client(host, port, operation, filename, file_data)
                worker_end = time.time()
                worker_results.append({
                    'worker_id': worker_id,
                    'success': success,
                    'time': worker_end - worker_start,
                    'data_size': len(file_data) if file_data else 0
                })
            except Exception as e:
                worker_end = time.time()
                worker_results.append({
                    'worker_id': worker_id,
                    'success': False,
                    'time': worker_end - worker_start,
                    'data_size': 0,
                    'error': str(e)
                })
        
        # Start worker threads
        logger.info(f"Starting {num_workers} worker threads for {operation} operation")
        for i in range(num_workers):
            thread = threading.Thread(target=worker_task, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=timeout)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Process results
        successful_workers = sum(1 for r in worker_results if r.get('success', False))
        failed_workers = num_workers - successful_workers
        
        if successful_workers > 0:
            avg_time = sum(r['time'] for r in worker_results if r.get('success', False)) / successful_workers
            total_data = sum(r['data_size'] for r in worker_results if r.get('success', False))
            throughput = total_data / total_time if total_time > 0 else 0
        else:
            avg_time = 0
            throughput = 0
        
        results.update({
            'successful_workers': successful_workers,
            'failed_workers': failed_workers,
            'total_time': total_time,
            'avg_time_per_client': avg_time,
            'throughput': throughput
        })
        
        logger.info(f"Test completed: {successful_workers} success, {failed_workers} failed")
        return results
    
    def run_quick_tests(self):
        """Run quick stress tests with automatic server management"""
        logger.info("=" * 60)
        logger.info("STARTING IMPROVED QUICK STRESS TEST")
        logger.info("=" * 60)
        
        # Test configurations
        configs = [
            {'op': 'upload', 'workers': 1, 'mode': 'thread', 'server_workers': 3},
            {'op': 'upload', 'workers': 5, 'mode': 'thread', 'server_workers': 3},
            {'op': 'download', 'workers': 1, 'mode': 'thread', 'server_workers': 3},
            {'op': 'download', 'workers': 5, 'mode': 'thread', 'server_workers': 3},
        ]
        
        # Create test file
        test_file_path = self.create_test_file("quick_test.bin", 10)  # 10MB
        with open(test_file_path, 'rb') as f:
            file_data = f.read()
        
        logger.info(f"Running {len(configs)} test configurations...")
        
        server_process = None
        server_port = None
        
        try:
            # Start server once for all tests
            server_process, server_port = self.start_server('thread', 3)
            
            for i, config in enumerate(configs, 1):
                logger.info(f"\n--- Test {i}/{len(configs)} ---")
                logger.info(f"Operation: {config['op']}, Workers: {config['workers']}, Mode: {config['mode']}")
                
                try:
                    # For download tests, upload file first
                    if config['op'] == 'download':
                        logger.info("Uploading test file for download test...")
                        upload_success = self.simple_file_client(
                            'localhost', server_port, 'upload', 
                            "quick_test.bin", file_data
                        )
                        if not upload_success:
                            logger.error("Failed to upload test file")
                            raise RuntimeError("Upload failed")
                    
                    # Run stress test
                    result = self.run_simple_stress_test(
                        config['op'],
                        'localhost',
                        server_port,
                        "quick_test.bin",
                        file_data,
                        config['workers'],
                        timeout=60
                    )
                    
                    # Create result row
                    row = {
                        'Nomor': i,
                        'Execution_Mode': config['mode'],
                        'Operasi': config['op'],
                        'Volume': 10,  # 10MB test file
                        'Jumlah_Client_Worker_Pool': config['workers'],
                        'Jumlah_Server_Worker_Pool': config['server_workers'],
                        'Waktu_Total_Per_Client': result['avg_time_per_client'],
                        'Throughput_Per_Client': result['throughput'],
                        'Client_Worker_Sukses': result['successful_workers'],
                        'Client_Worker_Gagal': result['failed_workers'],
                        'Throughput_MB_per_sec': result['throughput'] / (1024 * 1024),
                        'Success_Rate': result['successful_workers'] / config['workers'] if config['workers'] > 0 else 0,
                        'Total_Time': result['total_time']
                    }
                    
                    self.results.append(row)
                    
                    logger.info(f"✓ Result: {result['successful_workers']} success, {result['failed_workers']} failed")
                    logger.info(f"✓ Throughput: {result['throughput'] / (1024 * 1024):.2f} MB/s")
                    logger.info(f"✓ Avg time per client: {result['avg_time_per_client']:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Test {i} failed: {e}")
                    row = {
                        'Nomor': i,
                        'Execution_Mode': config['mode'],
                        'Operasi': config['op'],
                        'Volume': 10,
                        'Jumlah_Client_Worker_Pool': config['workers'],
                        'Jumlah_Server_Worker_Pool': config['server_workers'],
                        'Waktu_Total_Per_Client': 0,
                        'Throughput_Per_Client': 0,
                        'Client_Worker_Sukses': 0,
                        'Client_Worker_Gagal': config['workers'],
                        'Throughput_MB_per_sec': 0,
                        'Success_Rate': 0,
                        'Total_Time': 0,
                        'Error': str(e)
                    }
                    self.results.append(row)
                
                # Small delay between tests
                time.sleep(2)
        
        finally:
            # Stop server
            if server_process and server_port:
                self.stop_server(server_process, server_port)
    
    def save_results(self):
        """Save test results to CSV"""
        if not self.results:
            logger.warning("No results to save")
            return
        
        try:
            import pandas as pd
            
            df = pd.DataFrame(self.results)
            
            # Round numeric columns
            numeric_cols = ['Waktu_Total_Per_Client', 'Throughput_Per_Client', 
                           'Throughput_MB_per_sec', 'Success_Rate', 'Total_Time']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].round(4)
            
            # Save results
            output_file = 'improved_quick_test_results.csv'
            df.to_csv(output_file, index=False)
            
            logger.info(f"\n{'='*60}")
            logger.info("QUICK TEST RESULTS SUMMARY:")
            logger.info("="*60)
            
            # Print summary
            total_tests = len(df)
            successful_tests = len(df[df['Client_Worker_Gagal'] == 0])
            
            logger.info(f"Total tests run: {total_tests}")
            logger.info(f"Successful tests: {successful_tests}")
            logger.info(f"Failed tests: {total_tests - successful_tests}")
            logger.info(f"Success rate: {successful_tests/total_tests*100:.1f}%")
            
            if successful_tests > 0:
                avg_throughput = df[df['Client_Worker_Gagal'] == 0]['Throughput_MB_per_sec'].mean()
                max_throughput = df['Throughput_MB_per_sec'].max()
                logger.info(f"Average throughput: {avg_throughput:.2f} MB/s")
                logger.info(f"Maximum throughput: {max_throughput:.2f} MB/s")
            
            logger.info(f"\nDetailed results saved to: {output_file}")
            logger.info("\nResults table:")
            print(df.to_string(index=False))
            
        except ImportError:
            logger.warning("pandas not available, saving results as text")
            self.save_results_as_text()
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            self.save_results_as_text()
    
    def save_results_as_text(self):
        """Save results as plain text (fallback)"""
        output_file = 'improved_quick_test_results.txt'
        with open(output_file, 'w') as f:
            f.write("IMPROVED QUICK TEST RESULTS\n")
            f.write("="*50 + "\n\n")
            
            for i, result in enumerate(self.results, 1):
                f.write(f"Test {i}:\n")
                for key, value in result.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        logger.info(f"Results saved to: {output_file}")
    
    def cleanup(self):
        """Clean up test files and processes"""
        logger.info("Cleaning up...")
        
        # Stop any remaining server processes
        for port, process in list(self.server_processes.items()):
            self.stop_server(process, port)
        
        # Remove test files
        for file_path in self.cleanup_files:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Removed test file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
        
        logger.info("Cleanup completed")
    
    def run(self):
        """Main execution method"""
        try:
            logger.info("Improved Quick Test Runner Starting...")
            
            # Check dependencies
            if not self.check_dependencies():
                logger.error("Dependency check failed. Please install missing requirements.")
                return False
            
            # Run tests
            self.run_quick_tests()
            
            # Save results
            self.save_results()
            
            logger.info("\n" + "="*60)
            logger.info("✅ IMPROVED QUICK TEST COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            
            return True
            
        except KeyboardInterrupt:
            logger.info("\nTest interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return False
        finally:
            self.cleanup()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("\nReceived interrupt signal, cleaning up...")
    sys.exit(0)

def main():
    """Main entry point"""
    signal.signal(signal.SIGINT, signal_handler)
    
    print("="*60)
    print("IMPROVED QUICK TEST RUNNER")
    print("="*60)
    print("This test will run 4 configurations:")
    print("- Upload with 1 worker")
    print("- Upload with 5 workers") 
    print("- Download with 1 worker")
    print("- Download with 5 workers")
    print()
    print("Features:")
    print("✓ Automatic server management")
    print("✓ Port availability checking")
    print("✓ Comprehensive error handling")
    print("✓ Automatic cleanup")
    print("✓ Detailed reporting")
    print("="*60)
    print()
    
    runner = ImprovedQuickTestRunner()
    success = runner.run()
    
    if success:
        print("Test completed successfully!")
        sys.exit(0)
    else:
        print("Test failed or was interrupted.")
        sys.exit(1)

if __name__ == "__main__":
    main()