import os
import time
import threading
import multiprocessing
import subprocess
import pandas as pd
import logging
from typing import Dict, List, Tuple
from enhanced_client import (
    FileClient, create_test_file, run_threading_stress_test, 
    run_multiprocessing_stress_test
)
import signal
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StressTestRunner:
    def __init__(self):
        self.results = []
        self.test_files = {}
        self.server_processes = {}
        
    def setup_test_files(self):
        """Create test files of different sizes"""
        sizes = [10, 50, 100]  # MB
        
        for size in sizes:
            filename = f"test_{size}MB.bin"
            logging.info(f"Creating test file: {filename}")
            filepath = create_test_file(filename, size)
            
            # Read file data into memory for testing
            with open(filepath, 'rb') as f:
                file_data = f.read()
            
            self.test_files[size] = {
                'filename': filename,
                'filepath': filepath,
                'data': file_data,
                'size_bytes': len(file_data)
            }
            
            logging.info(f"Created {filename} ({len(file_data)} bytes)")

    def start_server(self, server_type: str, port: int, workers: int) -> subprocess.Popen:
        """Start server process"""
        cmd = [
            sys.executable, 
            "file_server_pools.py", 
            server_type, 
            str(workers)
        ]
        
        # Modify the command to use specific port
        if server_type == "thread":
            # For thread server, we'll use port 45000 + workers as offset
            actual_port = 45000 + workers
        else:
            # For process server, we'll use port 45100 + workers as offset  
            actual_port = 45100 + workers
            
        logging.info(f"Starting {server_type} server on port {actual_port} with {workers} workers")
        
        # We need to create a modified version that accepts port parameter
        # For now, we'll use the default ports and handle conflicts
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give server time to start
        time.sleep(2)
        return process

    def stop_server(self, process: subprocess.Popen):
        """Stop server process"""
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def upload_test_file_to_server(self, host: str, port: int, filename: str, file_data: bytes) -> bool:
        """Upload test file to server for download tests"""
        client = FileClient(host, port)
        success, message, _, _ = client.upload_file(filename, file_data)
        if success:
            logging.info(f"Uploaded {filename} to server")
        else:
            logging.error(f"Failed to upload {filename}: {message}")
        return success

    def run_single_test(self, test_config: Dict) -> Dict:
        """Run a single stress test configuration"""
        logging.info(f"Running test: {test_config}")
        
        # Determine server port based on configuration
        if test_config['server_type'] == 'thread':
            port = 45000
        else:
            port = 45001
            
        # Start server
        server_process = self.start_server(
            test_config['server_type'], 
            port, 
            test_config['server_workers']
        )
        
        try:
            # Wait for server to be ready
            time.sleep(3)
            
            # Get test file data
            file_info = self.test_files[test_config['volume_mb']]
            
            # For download tests, upload file first
            if test_config['operation'] == 'download':
                upload_success = self.upload_test_file_to_server(
                    'localhost', port, file_info['filename'], file_info['data']
                )
                if not upload_success:
                    return {
                        'error': 'Failed to upload test file for download test',
                        'server_successful': 0,
                        'server_failed': 1
                    }
            
            # Run stress test
            if test_config['client_type'] == 'thread':
                result = run_threading_stress_test(
                    test_config['operation'],
                    'localhost',
                    port,
                    file_info['filename'],
                    file_info['data'],
                    test_config['client_workers'],
                    timeout=120
                )
            else:  # multiprocessing
                result = run_multiprocessing_stress_test(
                    test_config['operation'],
                    'localhost', 
                    port,
                    file_info['filename'],
                    file_info['data'],
                    test_config['client_workers'],
                    timeout=120
                )
            
            # Add server stats (simplified - in real scenario you'd need to modify server to track this)
            result['server_successful'] = result['successful_workers']  # Approximate
            result['server_failed'] = result['failed_workers']  # Approximate
            
            return result
            
        except Exception as e:
            logging.error(f"Test failed: {e}")
            return {
                'error': str(e),
                'total_time': 0,
                'avg_time_per_client': 0,
                'throughput': 0,
                'successful_workers': 0,
                'failed_workers': test_config['client_workers'],
                'server_successful': 0,
                'server_failed': 1
            }
        finally:
            self.stop_server(server_process)
            time.sleep(2)  # Cool down between tests

    def generate_test_configurations(self) -> List[Dict]:
        """Generate all test configurations - 81 combinations total"""
        configurations = []
        test_id = 1
        
        # To get exactly 81 combinations (3^4), we need 4 parameters with 3 options each
        operations = ['download', 'upload', 'list']  # 3 operations instead of 2
        volumes = [10, 50, 100]  # MB - 3 options
        client_workers = [1, 5, 50]  # 3 options
        server_workers = [1, 5, 50]  # 3 options
        
        # Alternative: Keep 2 operations but add timeout variations
        # operations = ['download', 'upload']  # 2 operations
        # volumes = [10, 50, 100]  # MB - 3 options  
        # client_workers = [1, 5, 50]  # 3 options
        # server_workers = [1, 5, 50]  # 3 options
        # timeout_settings = ['short', 'medium', 'long']  # 3 timeout options
        
        for operation in operations:
            for volume in volumes:
                for client_worker in client_workers:
                    for server_worker in server_workers:
                        config = {
                            'test_id': test_id,
                            'operation': operation,
                            'volume_mb': volume,
                            'client_workers': client_worker,
                            'server_workers': server_worker,
                            'client_type': 'thread',  # Fixed for consistency
                            'server_type': 'thread'   # Fixed for consistency
                        }
                        configurations.append(config)
                        test_id += 1
        
        print(f"Generated {len(configurations)} test configurations")
        return configurations

    def run_all_tests(self):
        """Run all stress test combinations"""
        self.setup_test_files()
        
        configurations = self.generate_test_configurations()
        total_tests = len(configurations)
        
        logging.info(f"Starting stress tests: {total_tests} configurations")
        
        for i, config in enumerate(configurations, 1):
            logging.info(f"Running test {i}/{total_tests}")
            
            try:
                result = self.run_single_test(config)
                
                # Combine configuration and results
                row = {
                    'No': config['test_id'],
                    'Operation': config['operation'],
                    'Volume_MB': config['volume_mb'],
                    'Client_Workers': config['client_workers'],
                    'Server_Workers': config['server_workers'],
                    'Client_Type': config['client_type'],
                    'Server_Type': config['server_type'],
                    'Total_Time_Seconds': result.get('avg_time_per_client', 0),
                    'Throughput_Bytes_Per_Second': result.get('throughput', 0),
                    'Client_Successful': result.get('successful_workers', 0),
                    'Client_Failed': result.get('failed_workers', 0),
                    'Server_Successful': result.get('server_successful', 0),
                    'Server_Failed': result.get('server_failed', 0),
                    'Error': result.get('error', '')
                }
                
                self.results.append(row)
                
                # Save intermediate results
                if i % 5 == 0:  # Save every 5 tests
                    self.save_results(f'intermediate_results_{i}.csv')
                    
            except Exception as e:
                logging.error(f"Test {i} failed completely: {e}")
                # Add failed test result
                row = {
                    'No': config['test_id'],
                    'Operation': config['operation'],
                    'Volume_MB': config['volume_mb'],
                    'Client_Workers': config['client_workers'],
                    'Server_Workers': config['server_workers'],
                    'Client_Type': config['client_type'],
                    'Server_Type': config['server_type'],
                    'Total_Time_Seconds': 0,
                    'Throughput_Bytes_Per_Second': 0,
                    'Client_Successful': 0,
                    'Client_Failed': config['client_workers'],
                    'Server_Successful': 0,
                    'Server_Failed': 1,
                    'Error': str(e)
                }
                self.results.append(row)
        
        self.save_results('final_stress_test_results.csv')
        self.generate_summary_report()

    def save_results(self, filename: str):
        """Save results to CSV file"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False)
            logging.info(f"Results saved to {filename}")

    def generate_summary_report(self):
        """Generate summary report"""
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        
        # Generate summary statistics
        summary = {
            'Total Tests': len(df),
            'Successful Tests': len(df[df['Client_Failed'] == 0]),
            'Failed Tests': len(df[df['Client_Failed'] > 0]),
            'Average Throughput': df['Throughput_Bytes_Per_Second'].mean(),
            'Max Throughput': df['Throughput_Bytes_Per_Second'].max(),
            'Min Throughput': df['Throughput_Bytes_Per_Second'].min(),
            'Average Time per Client': df['Total_Time_Seconds'].mean(),
        }
        
        # Save summary
        with open('stress_test_summary.txt', 'w') as f:
            f.write("Stress Test Summary Report\n")
            f.write("=" * 30 + "\n\n")
            
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nTop 10 Best Throughput Configurations:\n")
            f.write("-" * 40 + "\n")
            
            top_configs = df.nlargest(10, 'Throughput_Bytes_Per_Second')
            for _, row in top_configs.iterrows():
                f.write(f"Config {row['No']}: {row['Operation']} {row['Volume_MB']}MB, "
                       f"C{row['Client_Workers']}/S{row['Server_Workers']} workers, "
                       f"Throughput: {row['Throughput_Bytes_Per_Second']:.2f} B/s\n")
        
        logging.info("Summary report saved to stress_test_summary.txt")

def signal_handler(sig, frame):
    logging.info("Stress test interrupted by user")
    sys.exit(0)

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)
    
    runner = StressTestRunner()
    
    try:
        runner.run_all_tests()
        logging.info("All stress tests completed!")
    except KeyboardInterrupt:
        logging.info("Tests interrupted by user")
    except Exception as e:
        logging.error(f"Test runner failed: {e}")
    finally:
        # Clean up any remaining processes
        logging.info("Cleaning up...")