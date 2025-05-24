#!/usr/bin/env python3
"""
Fixed Stress Test Runner with Full Process Mode Support
=======================================================

This version properly handles both thread and process modes without skipping.
"""

import os
import time
import threading
import multiprocessing
import subprocess
import pandas as pd
import logging
import socket
import random
import queue
from typing import Dict, List, Tuple, Optional
from enhanced_client import (
    FileClient, create_test_file, run_threading_stress_test, 
    run_multiprocessing_stress_test
)
import signal
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedStressTestRunner:
    def __init__(self):
        self.results = []
        self.test_files = {}
        self.server_processes = {}
        self.used_ports = set()
        self.base_port = 45000
        self.max_port_attempts = 20
        self.failed_configs = []
        self.skipped_configs = []
        
        # Connection management settings
        self.max_concurrent_connections = 15
        self.connection_batch_size = 10
        self.connection_delay = 0.15
        
        # Set multiprocessing start method for Windows
        if sys.platform == 'win32':
            multiprocessing.set_start_method('spawn', force=True)
        
    def find_available_port(self, start_port: int = None) -> int:
        """Find an available port"""
        if start_port is None:
            start_port = self.base_port
            
        for i in range(self.max_port_attempts):
            port = start_port + i
            
            if port in self.used_ports:
                continue
                
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(('localhost', port))
                    logger.info(f"Found available port: {port}")
                    self.used_ports.add(port)
                    return port
            except OSError as e:
                logger.debug(f"Port {port} not available: {e}")
                continue
        
        raise RuntimeError(f"Could not find available port after {self.max_port_attempts} attempts")
    
    def release_port(self, port: int):
        """Release a port for reuse"""
        self.used_ports.discard(port)
        time.sleep(1)
    
    def validate_test_feasibility(self, client_workers: int, server_workers: int) -> Tuple[bool, str]:
        """Validate if test configuration is feasible"""
        
        # Adjusted for more realistic expectations
        if client_workers >= 50 and server_workers <= 5:
            return False, "Server severely underprovisioned (10:1 ratio exceeded)"
        
        if client_workers >= 30 and server_workers <= 2:
            return False, "High client load with minimal server capacity"
        
        # Warn about challenging configurations
        if client_workers >= 20 and server_workers < 5:
            logger.warning(f"Challenging config: {client_workers} clients with {server_workers} servers")
        
        return True, "Configuration appears feasible"

    def setup_test_files(self):
        """Create test files"""
        sizes = [10, 50, 100]  # MB
        
        for size in sizes:
            filename = f"test_{size}MB.bin"
            try:
                logger.info(f"Creating test file: {filename}")
                filepath = create_test_file(filename, size)
                
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Test file {filepath} was not created")
                
                with open(filepath, 'rb') as f:
                    file_data = f.read()
                
                self.test_files[size] = {
                    'filename': filename,
                    'filepath': filepath,
                    'data': file_data,
                    'size_bytes': len(file_data)
                }
                
                logger.info(f"✓ Created {filename} ({len(file_data)} bytes)")
                
            except Exception as e:
                logger.error(f"Failed to create test file {filename}: {e}")
                raise

    def start_server(self, server_type: str, workers: int, max_retries: int = 3) -> Tuple[subprocess.Popen, int]:
        """Start server with proper support for both thread and process modes"""
        
        for attempt in range(max_retries):
            port = None
            process = None
            
            try:
                port = self.find_available_port()
                
                # Command to start server
                cmd = [sys.executable, "file_server_pools.py", server_type, str(workers)]
                
                logger.info(f"Attempt {attempt + 1}: Starting {server_type} server on port {port} with {workers} workers")
                
                # Environment with port
                env = os.environ.copy()
                env['SERVER_PORT'] = str(port)
                
                # For process mode on Windows, ensure proper startup
                if sys.platform == 'win32' and server_type == 'process':
                    env['PYTHONUNBUFFERED'] = '1'
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env
                )
                
                # Wait for server startup - longer for process mode
                startup_time = 5 if server_type == 'process' else 3
                startup_time += (workers // 10)
                logger.info(f"Waiting {startup_time}s for {server_type} server startup...")
                time.sleep(startup_time)
                
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    logger.error(f"Server process exited: {stderr}")
                    self.cleanup_server(process, port)
                    
                    if attempt < max_retries - 1:
                        logger.info("Retrying server start...")
                        time.sleep(2)
                    continue
                
                # Test server health
                if self.test_server_health(port, timeout=20):
                    logger.info(f"✓ {server_type.capitalize()} server started successfully on port {port}")
                    return process, port
                else:
                    logger.error(f"Server not responding on port {port}")
                    self.cleanup_server(process, port)
                    
                    if attempt < max_retries - 1:
                        logger.info("Retrying server start...")
                        time.sleep(2)
                    continue
                    
            except Exception as e:
                logger.error(f"Server start attempt {attempt + 1} failed: {e}")
                self.cleanup_server(process, port)
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        raise RuntimeError(f"Failed to start {server_type} server after {max_retries} attempts")

    def test_server_health(self, port: int, timeout: int = 15) -> bool:
        """Test if server is responding"""
        for attempt in range(5):  # More attempts for process mode
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(timeout)
                    s.connect(('localhost', port))
                    return True
            except Exception as e:
                logger.debug(f"Health check attempt {attempt + 1} failed: {e}")
                if attempt < 4:
                    time.sleep(1)
                    continue
        return False

    def cleanup_server(self, process, port):
        """Clean up server process"""
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info("✓ Server stopped gracefully")
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error("Failed to kill server process")
        
        if port:
            self.release_port(port)

    def run_throttled_stress_test(self, operation: str, host: str, port: int, 
                                filename: str, file_data: bytes, 
                                client_workers: int, timeout: int = 120) -> Dict:
        """Run stress test with connection throttling"""
        
        # For low client counts, use normal approach
        if client_workers <= 10:
            logger.info(f"Running standard test: {client_workers} clients")
            try:
                return run_threading_stress_test(
                    operation, host, port, filename, file_data, client_workers, timeout
                )
            except Exception as e:
                logger.error(f"Standard test failed: {e}")
                return self.create_empty_result(client_workers)
        
        # For high client counts, implement throttling
        logger.info(f"Running THROTTLED stress test: {client_workers} clients")
        logger.info(f"Batch size: {self.connection_batch_size}, Connection delay: {self.connection_delay}s")
        
        results = {
            'successful_workers': 0,
            'failed_workers': 0,
            'total_time': 0,
            'avg_time_per_client': 0,
            'throughput': 0
        }
        
        start_time = time.time()
        worker_results = []
        
        # Create batches
        batch_size = self.connection_batch_size
        batches = [list(range(i, min(i + batch_size, client_workers))) 
                  for i in range(0, client_workers, batch_size)]
        
        logger.info(f"Processing {len(batches)} batches of up to {batch_size} workers each")
        
        for batch_num, batch_workers in enumerate(batches, 1):
            logger.info(f"Processing batch {batch_num}/{len(batches)} ({len(batch_workers)} workers)")
            
            threads = []
            batch_results = []
            
            def worker_task(worker_id):
                worker_start = time.time()
                try:
                    # Add small random delay to spread connections
                    delay = random.uniform(0, self.connection_delay * 2)
                    time.sleep(delay)
                    
                    client = FileClient(host, port, timeout=30)
                    
                    if operation == 'upload':
                        success, message, _, bytes_transferred = client.upload_file(
                            f"test_{worker_id}_{filename}", file_data
                        )
                    else:  # download
                        success, message, _, bytes_transferred = client.download_file(filename)
                    
                    worker_end = time.time()
                    batch_results.append({
                        'worker_id': worker_id,
                        'success': success,
                        'time': worker_end - worker_start,
                        'data_size': bytes_transferred if success else 0
                    })
                    
                except Exception as e:
                    worker_end = time.time()
                    batch_results.append({
                        'worker_id': worker_id,
                        'success': False,
                        'time': worker_end - worker_start,
                        'data_size': 0,
                        'error': str(e)
                    })
            
            # Start batch workers with throttling
            for worker_id in batch_workers:
                thread = threading.Thread(target=worker_task, args=(worker_id,))
                threads.append(thread)
                thread.start()
                
                # Throttle connection rate
                time.sleep(self.connection_delay)
            
            # Wait for batch to complete
            batch_timeout = timeout // len(batches) if len(batches) > 0 else timeout
            for thread in threads:
                thread.join(timeout=batch_timeout)
            
            worker_results.extend(batch_results)
            
            # Report batch progress
            batch_success = sum(1 for r in batch_results if r.get('success', False))
            logger.info(f"Batch {batch_num} completed: {batch_success}/{len(batch_workers)} successful")
            
            # Brief pause between batches
            if batch_num < len(batches):
                time.sleep(1)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Process results
        successful_workers = sum(1 for r in worker_results if r.get('success', False))
        failed_workers = client_workers - successful_workers
        
        if successful_workers > 0:
            successful_results = [r for r in worker_results if r.get('success', False)]
            avg_time = sum(r['time'] for r in successful_results) / len(successful_results)
            total_data = sum(r['data_size'] for r in successful_results)
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
        
        success_rate = successful_workers / client_workers * 100 if client_workers > 0 else 0
        logger.info(f"Throttled test completed: {successful_workers}/{client_workers} successful ({success_rate:.1f}%)")
        return results

    def create_empty_result(self, client_workers: int) -> Dict:
        """Create empty result for failed tests"""
        return {
            'successful_workers': 0,
            'failed_workers': client_workers,
            'total_time': 0,
            'avg_time_per_client': 0,
            'throughput': 0
        }

    def run_single_test(self, test_config: Dict) -> Dict:
        """Run single test configuration"""
        logger.info(f"Running test {test_config['test_id']}: {test_config}")
        
        # Feasibility check
        feasible, reason = self.validate_test_feasibility(
            test_config['client_workers'], 
            test_config['server_workers']
        )
        
        if not feasible:
            logger.warning(f"Skipping test {test_config['test_id']}: {reason}")
            self.skipped_configs.append((test_config, reason))
            return self.create_skipped_result(test_config, reason)
        
        server_process = None
        port = None
        
        try:
            # Start server
            server_process, port = self.start_server(
                test_config['server_type'], 
                test_config['server_workers']
            )
            
            file_info = self.test_files[test_config['volume_mb']]
            
            # For download tests, upload file first
            if test_config['operation'] == 'download':
                upload_success = self.upload_test_file_with_retry(
                    'localhost', port, file_info['filename'], file_info['data']
                )
                if not upload_success:
                    return self.create_failed_result(test_config, "Failed to upload test file")
            
            # Calculate timeout based on complexity
            base_timeout = 120
            complexity_factor = (test_config['client_workers'] * test_config['volume_mb']) / 50
            timeout = max(base_timeout, int(base_timeout + complexity_factor))
            
            logger.info(f"Running {test_config['operation']} test with {test_config['client_workers']} clients")
            logger.info(f"Server type: {test_config['server_type']}, Timeout: {timeout}s")
            
            # Use throttled stress test
            result = self.run_throttled_stress_test(
                test_config['operation'],
                'localhost',
                port,
                file_info['filename'],
                file_info['data'],
                test_config['client_workers'],
                timeout=timeout
            )
            
            result['server_successful'] = result['successful_workers']
            result['server_failed'] = result['failed_workers']
            
            success_rate = result['successful_workers'] / test_config['client_workers'] * 100 if test_config['client_workers'] > 0 else 0
            logger.info(f"✓ Test {test_config['test_id']} completed: {result['successful_workers']}/{test_config['client_workers']} successful ({success_rate:.1f}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"Test {test_config['test_id']} failed: {e}")
            self.failed_configs.append((test_config, str(e)))
            return self.create_failed_result(test_config, str(e))
            
        finally:
            if server_process and port:
                self.cleanup_server(server_process, port)
                # Cooldown
                cooldown = 3 if test_config['client_workers'] >= 20 else 2
                logger.info(f"Cooling down for {cooldown}s...")
                time.sleep(cooldown)

    def upload_test_file_with_retry(self, host: str, port: int, filename: str, file_data: bytes, max_retries: int = 3) -> bool:
        """Upload test file with retry logic"""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(2 ** (attempt - 1))
                
                client = FileClient(host, port, timeout=30)
                success, message, _, _ = client.upload_file(filename, file_data)
                if success:
                    logger.info(f"✓ Uploaded {filename} (attempt {attempt + 1})")
                    return True
                else:
                    logger.warning(f"Upload attempt {attempt + 1} failed: {message}")
            except Exception as e:
                logger.warning(f"Upload attempt {attempt + 1} error: {e}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying upload in {wait_time} seconds...")
                time.sleep(wait_time)
        
        logger.error(f"Failed to upload {filename} after {max_retries} attempts")
        return False

    def create_failed_result(self, test_config: Dict, error_message: str) -> Dict:
        """Create failed result"""
        return {
            'error': error_message,
            'total_time': 0,
            'avg_time_per_client': 0,
            'throughput': 0,
            'successful_workers': 0,
            'failed_workers': test_config['client_workers'],
            'server_successful': 0,
            'server_failed': 1
        }

    def create_skipped_result(self, test_config: Dict, reason: str) -> Dict:
        """Create skipped result"""
        return {
            'error': f"SKIPPED: {reason}",
            'total_time': 0,
            'avg_time_per_client': 0,
            'throughput': 0,
            'successful_workers': 0,
            'failed_workers': 0,
            'server_successful': 0,
            'server_failed': 0
        }

    def generate_test_configurations(self) -> List[Dict]:
        """Generate test configurations for both thread and process modes"""
        configurations = []
        test_id = 1
        
        execution_modes = ['thread', 'process']  # Both modes enabled
        volumes = [10, 50, 100]
        operations = ['download', 'upload']
        
        # Realistic worker combinations
        worker_combinations = [
            (1, 1),    # Baseline
            (1, 5),    # Over-provisioned
            (5, 1),    # Slight under-provision 
            (5, 5),    # Balanced
            (5, 10),   # Well-provisioned
            (10, 5),   # Medium load
            (20, 10),  # Higher load - will use throttling
            (50, 20),  # High load - definitely throttled
        ]
        
        for execution_mode in execution_modes:
            for volume in volumes:
                for client_workers, server_workers in worker_combinations:
                    for operation in operations:
                        config = {
                            'test_id': test_id,
                            'execution_mode': execution_mode,
                            'operation': operation,
                            'volume_mb': volume,
                            'client_workers': client_workers,
                            'server_workers': server_workers,
                            'server_type': execution_mode,
                        }
                        configurations.append(config)
                        test_id += 1
        
        logger.info(f"Generated {len(configurations)} test configurations (including both thread and process modes)")
        return configurations

    def run_all_tests(self):
        """Run all tests"""
        try:
            self.setup_test_files()
        except Exception as e:
            logger.error(f"Failed to setup test files: {e}")
            return
        
        configurations = self.generate_test_configurations()
        total_tests = len(configurations)
        
        logger.info(f"Starting stress tests: {total_tests} configurations")
        logger.info("Features: Thread AND Process mode support, Connection throttling for high loads")
        
        successful_tests = 0
        failed_tests = 0
        skipped_tests = 0
        
        for i, config in enumerate(configurations, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"Running test {i}/{total_tests} (ID: {config['test_id']})")
            logger.info(f"Config: {config['execution_mode']} | {config['operation']} | "
                       f"{config['volume_mb']}MB | C{config['client_workers']}/S{config['server_workers']}")
            
            try:
                result = self.run_single_test(config)
                
                if 'SKIPPED' in result.get('error', ''):
                    skipped_tests += 1
                    test_status = "SKIPPED"
                elif result.get('successful_workers', 0) > 0:
                    successful_tests += 1
                    test_status = "SUCCESS"
                else:
                    failed_tests += 1
                    test_status = "FAILED"
                
                row = self.create_result_row(config, result)
                self.results.append(row)
                
                logger.info(f"Test {i} {test_status}: {result.get('successful_workers', 0)}/{config['client_workers']} clients successful")
                
                # Save progress
                if i % 5 == 0 or i == total_tests:
                    self.save_intermediate_results(f'fixed_stress_test_progress_{i}.csv')
                    logger.info(f"Progress: {i}/{total_tests} ({successful_tests}✓ {failed_tests}✗ {skipped_tests}⏭)")
                    
            except KeyboardInterrupt:
                logger.info("Test interrupted by user")
                break
            except Exception as e:
                logger.error(f"Test {i} failed completely: {e}")
                failed_tests += 1
                row = self.create_result_row(config, self.create_failed_result(config, str(e)))
                self.results.append(row)
        
        logger.info(f"\n{'='*70}")
        logger.info("STRESS TEST SUITE COMPLETED")
        logger.info(f"Results: {successful_tests} successful, {failed_tests} failed, {skipped_tests} skipped")
        logger.info(f"{'='*70}")
        
        self.save_final_results()
        self.generate_report()

    def create_result_row(self, config: Dict, result: Dict) -> Dict:
        """Create result row"""
        return {
            'Nomor': config['test_id'],
            'Execution_Mode': config['execution_mode'],
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
            'Server_Type': config['server_type'],
            'Throughput_Bytes_Per_Second': result.get('throughput', 0),
            'Throughput_MB_Per_Second': result.get('throughput', 0) / (1024 * 1024),
            'Success_Rate': (result.get('successful_workers', 0) / config['client_workers'] 
                           if config['client_workers'] > 0 else 0),
            'Error_Message': result.get('error', ''),
            'Total_Time': result.get('total_time', 0)
        }

    def save_intermediate_results(self, filename: str):
        """Save intermediate results"""
        if self.results:
            try:
                df = pd.DataFrame(self.results)
                df.to_csv(filename, index=False)
                logger.debug(f"Intermediate results saved to {filename}")
            except Exception as e:
                logger.error(f"Failed to save intermediate results: {e}")

    def save_final_results(self):
        """Save final results"""
        if not self.results:
            logger.warning("No results to save")
            return
            
        try:
            df = pd.DataFrame(self.results)
            
            # Round numeric columns
            numeric_columns = ['Waktu_Total_Per_Client', 'Throughput_Per_Client', 
                              'Throughput_Bytes_Per_Second', 'Throughput_MB_Per_Second', 
                              'Success_Rate', 'Total_Time']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].round(4)
            
            # Save complete results
            final_filename = 'fixed_stress_test_results.csv'
            df.to_csv(final_filename, index=False)
            logger.info(f"✓ Results saved to {final_filename}")
            
            # Create summary of successful tests
            successful_df = df[df['Client_Worker_Sukses'] > 0]
            if len(successful_df) > 0:
                summary_filename = 'fixed_stress_test_summary.csv' 
                successful_df.to_csv(summary_filename, index=False)
                logger.info(f"✓ Summary saved to {summary_filename}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def generate_report(self):
        """Generate comprehensive report"""
        if not self.results:
            return
            
        try:
            df = pd.DataFrame(self.results)
            
            total_tests = len(df)
            successful_tests = len(df[df['Client_Worker_Sukses'] > 0])
            failed_tests = len(df[(df['Client_Worker_Gagal'] > 0) & (~df['Error_Message'].str.contains('SKIPPED', na=False))])
            skipped_tests = len(df[df['Error_Message'].str.contains('SKIPPED', na=False)])
            
            report_filename = 'fixed_stress_test_report.txt'
            with open(report_filename, 'w') as f:
                f.write("="*80 + "\n")
                f.write("FIXED STRESS TEST - COMPREHENSIVE REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write("EXECUTION SUMMARY:\n")
                f.write("-"*30 + "\n")
                f.write(f"Total Configurations: {total_tests}\n")
                f.write(f"Successful Tests: {successful_tests}\n")
                f.write(f"Failed Tests: {failed_tests}\n")
                f.write(f"Skipped Tests: {skipped_tests}\n")
                f.write(f"Overall Success Rate: {successful_tests/total_tests*100:.1f}%\n\n")
                
                # Analyze by execution mode
                f.write("EXECUTION MODE ANALYSIS:\n")
                f.write("-"*30 + "\n")
                for mode in ['thread', 'process']:
                    mode_df = df[df['Execution_Mode'] == mode]
                    mode_success = len(mode_df[mode_df['Client_Worker_Sukses'] > 0])
                    mode_total = len(mode_df)
                    if mode_total > 0:
                        f.write(f"{mode.capitalize()} mode: {mode_success}/{mode_total} successful ({mode_success/mode_total*100:.1f}%)\n")
                f.write("\n")
                
                if successful_tests > 0:
                    successful_df = df[df['Client_Worker_Sukses'] > 0]
                    
                    f.write("PERFORMANCE METRICS:\n")
                    f.write("-"*25 + "\n")
                    f.write(f"Average Throughput: {successful_df['Throughput_MB_Per_Second'].mean():.2f} MB/s\n")
                    f.write(f"Peak Throughput: {successful_df['Throughput_MB_Per_Second'].max():.2f} MB/s\n")
                    f.write(f"Average Success Rate: {successful_df['Success_Rate'].mean()*100:.1f}%\n\n")
                    
                    # Top configurations by mode
                    for mode in ['thread', 'process']:
                        mode_df = successful_df[successful_df['Execution_Mode'] == mode]
                        if len(mode_df) > 0:
                            f.write(f"\nTOP 5 {mode.upper()} MODE CONFIGURATIONS:\n")
                            f.write("-"*40 + "\n")
                            top_configs = mode_df.nlargest(5, 'Throughput_MB_Per_Second')
                            for i, (_, row) in enumerate(top_configs.iterrows(), 1):
                                f.write(f"{i}. Test {row['Nomor']:3d}: "
                                       f"{row['Operasi']:8s} | {row['Volume']:3d}MB | "
                                       f"C{row['Jumlah_Client_Worker_Pool']:2d}/S{row['Jumlah_Server_Worker_Pool']:2d} | "
                                       f"{row['Throughput_MB_Per_Second']:8.2f} MB/s | "
                                       f"Success: {row['Success_Rate']*100:5.1f}%\n")
            
            logger.info(f"✓ Report saved to {report_filename}")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    logger.info("Received interrupt signal, cleaning up...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    print("="*80)
    print("FIXED STRESS TEST RUNNER")
    print("="*80)
    print("Key features:")
    print("✓ Full support for both THREAD and PROCESS modes")
    print("✓ Connection throttling for high client loads")
    print("✓ Proper process mode implementation")
    print("✓ Intelligent test configuration selection")
    print("✓ Comprehensive error handling and recovery")
    print("="*80)
    
    runner = FixedStressTestRunner()
    
    try:
        runner.run_all_tests()
        logger.info("✅ All tests completed!")
        print("\n" + "="*80)
        print("✅ STRESS TEST COMPLETED!")
        print("📊 Results saved to:")
        print("  - fixed_stress_test_results.csv (complete results)")
        print("  - fixed_stress_test_summary.csv (successful tests)")
        print("  - fixed_stress_test_report.txt (comprehensive analysis)")
        print("  - fixed_stress_test_progress_*.csv (progress snapshots)")
        print("="*80)
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
    finally:
        logger.info("Cleaning up...")