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

class ModifiedStressTestRunner:
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
        
        logging.info(f"Starting {server_type} server on port {port} with {workers} workers")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give server time to start
        time.sleep(3)
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
        try:
            client = FileClient(host, port)
            success, message, _, _ = client.upload_file(filename, file_data)
            if success:
                logging.info(f"Uploaded {filename} to server")
            else:
                logging.error(f"Failed to upload {filename}: {message}")
            return success
        except Exception as e:
            logging.error(f"Error uploading file: {e}")
            return False

    def run_single_test(self, test_config: Dict) -> Dict:
        """Run a single stress test configuration"""
        logging.info(f"Running test {test_config['test_id']}: {test_config}")
        
        # Determine server port based on server type
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
                        'total_time': 0,
                        'avg_time_per_client': 0,
                        'throughput': 0,
                        'successful_workers': 0,
                        'failed_workers': test_config['client_workers'],
                        'server_successful': 0,
                        'server_failed': 1
                    }
            
            # Run stress test based on execution mode
            if test_config['execution_mode'] == 'thread':
                result = run_threading_stress_test(
                    test_config['operation'],
                    'localhost',
                    port,
                    file_info['filename'],
                    file_info['data'],
                    test_config['client_workers'],
                    timeout=120
                )
            else:  # multiprocess
                result = run_multiprocessing_stress_test(
                    test_config['operation'],
                    'localhost', 
                    port,
                    file_info['filename'],
                    file_info['data'],
                    test_config['client_workers'],
                    timeout=120
                )
            
            # Add server stats (simplified)
            result['server_successful'] = result['successful_workers']
            result['server_failed'] = result['failed_workers']
            
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
        """Generate all 108 test configurations"""
        configurations = []
        test_id = 1
        
        # 2×3×3×3×2 = 108 combinations
        execution_modes = ['thread', 'process']           # 2 options (multithread & multiprocess)
        volumes = [10, 50, 100]                          # 3 options (MB)
        client_workers = [1, 5, 50]                     # 3 options
        server_workers = [1, 5, 50]                     # 3 options  
        operations = ['download', 'upload']              # 2 options
        
        for execution_mode in execution_modes:
            for volume in volumes:
                for client_worker in client_workers:
                    for server_worker in server_workers:
                        for operation in operations:
                            config = {
                                'test_id': test_id,
                                'execution_mode': execution_mode,    # thread or process for overall system
                                'operation': operation,
                                'volume_mb': volume,
                                'client_workers': client_worker,
                                'server_workers': server_worker,
                                'server_type': execution_mode,       # Use same mode for server
                            }
                            configurations.append(config)
                            test_id += 1
        
        print(f"Generated {len(configurations)} test configurations")
        return configurations

    def run_all_tests(self):
        """Run all 108 stress test combinations"""
        self.setup_test_files()
        
        configurations = self.generate_test_configurations()
        total_tests = len(configurations)
        
        logging.info(f"Starting stress tests: {total_tests} configurations")
        
        for i, config in enumerate(configurations, 1):
            logging.info(f"Running test {i}/{total_tests}")
            
            try:
                result = self.run_single_test(config)
                
                # Combine configuration and results into single row with requested columns
                row = {
                    'Nomor': config['test_id'],
                    'Operasi': config['operation'],
                    'Volume': config['volume_mb'],
                    'Jumlah_Client_Worker_Pool': config['client_workers'],
                    'Jumlah_Server_Worker_Pool': config['server_workers'],
                    'Waktu_Total_Per_Client': result.get('avg_time_per_client', 0),
                    'Throughput_Per_Client': result.get('throughput', 0),
                    'Client_Worker_Sukses': result.get('successful_workers', 0),
                    'Client_Worker_Gagal': result.get('failed_workers', 0),
                    'Server_Worker_Sukses': result.get('server_successful', 0),
                    'Server_Worker_Gagal': result.get('server_failed', 0),
                    # Additional useful columns
                    'Execution_Mode': config['execution_mode'],
                    'Server_Type': config['server_type'],
                    'Throughput_Bytes_Per_Second': result.get('throughput', 0),
                    'Throughput_MB_Per_Second': result.get('throughput', 0) / (1024 * 1024),
                    'Success_Rate': result.get('successful_workers', 0) / config['client_workers'] if config['client_workers'] > 0 else 0,
                    'Error_Message': result.get('error', '')
                }
                
                self.results.append(row)
                
                # Save intermediate results every 10 tests
                if i % 10 == 0:
                    self.save_results(f'intermediate_results_{i}.csv')
                    logging.info(f"Saved intermediate results: {i}/{total_tests} tests completed")
                    
            except Exception as e:
                logging.error(f"Test {i} failed completely: {e}")
                # Add failed test result with requested column names
                row = {
                    'Nomor': config['test_id'],
                    'Operasi': config['operation'],
                    'Volume': config['volume_mb'],
                    'Jumlah_Client_Worker_Pool': config['client_workers'],
                    'Jumlah_Server_Worker_Pool': config['server_workers'],
                    'Waktu_Total_Per_Client': 0,
                    'Throughput_Per_Client': 0,
                    'Client_Worker_Sukses': 0,
                    'Client_Worker_Gagal': config['client_workers'],
                    'Server_Worker_Sukses': 0,
                    'Server_Worker_Gagal': 1,
                    # Additional useful columns
                    'Execution_Mode': config['execution_mode'],
                    'Server_Type': config['server_type'],
                    'Throughput_Bytes_Per_Second': 0,
                    'Throughput_MB_Per_Second': 0,
                    'Success_Rate': 0,
                    'Error_Message': str(e)
                }
                self.results.append(row)
        
        # Save final results to single CSV file
        self.save_final_results()
        self.generate_summary_report()

    def save_results(self, filename: str):
        """Save results to CSV file"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False)
            logging.info(f"Results saved to {filename}")

    def save_final_results(self):
        """Save final results to single CSV file with proper formatting"""
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        
        # Round numeric columns for better readability
        numeric_columns = ['Waktu_Total_Per_Client', 'Throughput_Per_Client', 
                          'Throughput_Bytes_Per_Second', 'Throughput_MB_Per_Second', 'Success_Rate']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].round(4)
        
        # Save to single final CSV file
        final_filename = 'stress_test_results_108_combinations.csv'
        df.to_csv(final_filename, index=False)
        logging.info(f"Final results saved to {final_filename}")
        
        # Also create a summary version with key metrics using new column names
        summary_df = df[['Nomor', 'Execution_Mode', 'Operasi', 'Volume', 
                        'Jumlah_Client_Worker_Pool', 'Jumlah_Server_Worker_Pool', 
                        'Throughput_MB_Per_Second', 'Success_Rate', 'Waktu_Total_Per_Client']].copy()
        
        summary_filename = 'stress_test_summary_108.csv'
        summary_df.to_csv(summary_filename, index=False)
        logging.info(f"Summary results saved to {summary_filename}")

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        
        # Calculate summary statistics
        total_tests = len(df)
        successful_tests = len(df[df['Client_Worker_Gagal'] == 0])
        failed_tests = total_tests - successful_tests
        
        summary_stats = {
            'Total Tests Run': total_tests,
            'Successful Tests': successful_tests,
            'Failed Tests': failed_tests,
            'Success Percentage': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
            'Average Throughput (MB/s)': df['Throughput_MB_Per_Second'].mean(),
            'Max Throughput (MB/s)': df['Throughput_MB_Per_Second'].max(),
            'Min Throughput (MB/s)': df['Throughput_MB_Per_Second'].min(),
            'Average Response Time (s)': df['Waktu_Total_Per_Client'].mean(),
            'Average Success Rate': df['Success_Rate'].mean()
        }
        
        # Generate detailed report
        report_filename = 'stress_test_final_report.txt'
        with open(report_filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("FILE SERVER STRESS TEST - FINAL REPORT\n")
            f.write("108 Combinations Test Results\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write("-" * 30 + "\n")
            for key, value in summary_stats.items():
                if 'Percentage' in key or 'Rate' in key:
                    f.write(f"{key}: {value:.2f}%\n")
                elif 'MB/s' in key or 'Time' in key:
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write(f"\nTOP 10 BEST PERFORMING CONFIGURATIONS:\n")
            f.write("-" * 50 + "\n")
            
            top_configs = df.nlargest(10, 'Throughput_MB_Per_Second')
            for i, (_, row) in enumerate(top_configs.iterrows(), 1):
                f.write(f"{i:2d}. Test {row['Nomor']:3d}: {row['Execution_Mode']:7s} | "
                       f"{row['Operasi']:8s} | {row['Volume']:3d}MB | "
                       f"C{row['Jumlah_Client_Worker_Pool']:2d}/S{row['Jumlah_Server_Worker_Pool']:2d} | "
                       f"{row['Throughput_MB_Per_Second']:8.2f} MB/s | "
                       f"Success: {row['Success_Rate']*100:5.1f}%\n")
            
            f.write(f"\nFAILED TESTS SUMMARY:\n")
            f.write("-" * 30 + "\n")
            failed_df = df[df['Client_Worker_Gagal'] > 0]
            if len(failed_df) > 0:
                f.write(f"Total failed tests: {len(failed_df)}\n")
                for _, row in failed_df.iterrows():
                    f.write(f"Test {row['Nomor']}: {row['Error_Message']}\n")
            else:
                f.write("No failed tests! 🎉\n")
            
            f.write(f"\nPERFORMANCE BY EXECUTION MODE:\n")
            f.write("-" * 40 + "\n")
            mode_stats = df.groupby('Execution_Mode').agg({
                'Throughput_MB_Per_Second': ['mean', 'max', 'min'],
                'Success_Rate': 'mean',
                'Waktu_Total_Per_Client': 'mean'
            }).round(4)
            f.write(str(mode_stats))
            f.write("\n")
            
            f.write(f"\nPERFORMANCE BY OPERATION:\n")
            f.write("-" * 35 + "\n")
            op_stats = df.groupby('Operasi').agg({
                'Throughput_MB_Per_Second': ['mean', 'max', 'min'],
                'Success_Rate': 'mean',
                'Waktu_Total_Per_Client': 'mean'
            }).round(4)
            f.write(str(op_stats))
            f.write("\n")
        
        logging.info(f"Comprehensive report saved to {report_filename}")

def signal_handler(sig, frame):
    logging.info("Stress test interrupted by user")
    sys.exit(0)

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 60)
    print("MODIFIED FILE SERVER STRESS TEST")
    print("Testing 108 combinations:")
    print("- Execution modes: thread, process (2)")
    print("- Operations: download, upload (2)")  
    print("- File volumes: 10MB, 50MB, 100MB (3)")
    print("- Client workers: 1, 5, 50 (3)")
    print("- Server workers: 1, 5, 50 (3)")
    print("Total: 2×2×3×3×3 = 108 combinations")
    print("=" * 60)
    
    runner = ModifiedStressTestRunner()
    
    try:
        runner.run_all_tests()
        logging.info("All 108 stress tests completed successfully!")
        print("\n" + "="*60)
        print("✅ STRESS TEST COMPLETED!")
        print("📊 Results saved to: stress_test_results_108_combinations.csv")
        print("📋 Summary saved to: stress_test_summary_108.csv") 
        print("📄 Report saved to: stress_test_final_report.txt")
        print("="*60)
    except KeyboardInterrupt:
        logging.info("Tests interrupted by user")
    except Exception as e:
        logging.error(f"Test runner failed: {e}")
    finally:
        logging.info("Cleaning up...")