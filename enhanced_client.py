import socket
import base64
import time
import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import threading
from typing import Tuple, List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileClient:
    def __init__(self, host='localhost', port=45000, timeout=30):
        self.host = host
        self.port = port
        self.timeout = timeout

    def send_request(self, command: str) -> Tuple[bool, str, float]:
        """Send request to server and return (success, response, time_taken)"""
        start_time = time.time()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))
            
            # Send command
            sock.sendall(f"{command}\r\n".encode('utf-8'))
            
            # Receive response
            response_data = b""
            while True:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                response_data += chunk
                if b"\r\n\r\n" in response_data:
                    break
            
            response = response_data.decode('utf-8').replace('\r\n\r\n', '')
            end_time = time.time()
            
            sock.close()
            return True, response, end_time - start_time
            
        except Exception as e:
            end_time = time.time()
            logging.error(f"Request failed: {e}")
            return False, str(e), end_time - start_time

    def list_files(self) -> Tuple[bool, List[str], float]:
        """List files on server"""
        success, response, time_taken = self.send_request("LIST")
        if success:
            try:
                data = json.loads(response)
                if data['status'] == 'OK':
                    return True, data['data'], time_taken
                else:
                    return False, [data['data']], time_taken
            except json.JSONDecodeError:
                return False, [response], time_taken
        return False, [response], time_taken

    def upload_file(self, filename: str, file_data: bytes = None) -> Tuple[bool, str, float, int]:
        """Upload file to server, returns (success, message, time_taken, bytes_transferred)"""
        try:
            if file_data is None:
                if not os.path.exists(filename):
                    return False, f"File {filename} not found", 0, 0
                with open(filename, 'rb') as f:
                    file_data = f.read()
            
            encoded_data = base64.b64encode(file_data).decode('utf-8')
            command = f"UPLOAD {os.path.basename(filename)} {encoded_data}"
            
            success, response, time_taken = self.send_request(command)
            bytes_transferred = len(file_data) if success else 0
            
            return success, response, time_taken, bytes_transferred
            
        except Exception as e:
            return False, str(e), 0, 0

    def download_file(self, filename: str, save_path: str = None) -> Tuple[bool, str, float, int]:
        """Download file from server, returns (success, message, time_taken, bytes_transferred)"""
        success, response, time_taken = self.send_request(f"GET {filename}")
        
        if success:
            try:
                data = json.loads(response)
                if data['status'] == 'OK':
                    file_content = base64.b64decode(data['data_file'])
                    bytes_transferred = len(file_content)
                    
                    if save_path:
                        with open(save_path, 'wb') as f:
                            f.write(file_content)
                    
                    return True, f"Downloaded {filename} ({bytes_transferred} bytes)", time_taken, bytes_transferred
                else:
                    return False, data['data'], time_taken, 0
            except (json.JSONDecodeError, KeyError) as e:
                return False, f"Invalid response format: {e}", time_taken, 0
        
        return False, response, time_taken, 0

def create_test_file(filename: str, size_mb: int):
    """Create test file of specified size"""
    if not os.path.exists('test_files'):
        os.makedirs('test_files')
    
    filepath = os.path.join('test_files', filename)
    with open(filepath, 'wb') as f:
        # Write data in chunks to avoid memory issues
        chunk_size = 1024 * 1024  # 1MB chunks
        data_chunk = b'A' * chunk_size
        
        for _ in range(size_mb):
            f.write(data_chunk)
    
    return filepath

def worker_upload_test(args) -> Dict:
    """Worker function for upload stress test"""
    worker_id, host, port, filename, file_data, timeout = args
    client = FileClient(host, port, timeout)
    
    start_time = time.time()
    success, message, request_time, bytes_transferred = client.upload_file(f"test_{worker_id}_{filename}", file_data)
    total_time = time.time() - start_time
    
    throughput = bytes_transferred / total_time if total_time > 0 else 0
    
    return {
        'worker_id': worker_id,
        'success': success,
        'message': message,
        'total_time': total_time,
        'request_time': request_time,
        'bytes_transferred': bytes_transferred,
        'throughput': throughput
    }

def worker_download_test(args) -> Dict:
    """Worker function for download stress test"""
    worker_id, host, port, filename, timeout = args
    client = FileClient(host, port, timeout)
    
    start_time = time.time()
    success, message, request_time, bytes_transferred = client.download_file(filename)
    total_time = time.time() - start_time
    
    throughput = bytes_transferred / total_time if total_time > 0 else 0
    
    return {
        'worker_id': worker_id,
        'success': success,
        'message': message,
        'total_time': total_time,
        'request_time': request_time,
        'bytes_transferred': bytes_transferred,
        'throughput': throughput
    }

def run_threading_stress_test(operation: str, host: str, port: int, filename: str, 
                            file_data: bytes, num_workers: int, timeout: int = 60) -> Dict:
    """Run stress test using ThreadPoolExecutor"""
    
    if operation == 'upload':
        worker_args = [(i, host, port, filename, file_data, timeout) for i in range(num_workers)]
        worker_func = worker_upload_test
    else:  # download
        worker_args = [(i, host, port, filename, timeout) for i in range(num_workers)]
        worker_func = worker_download_test
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_func, args) for args in worker_args]
        
        for future in futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                results.append({
                    'worker_id': -1,
                    'success': False,
                    'message': str(e),
                    'total_time': 0,
                    'request_time': 0,
                    'bytes_transferred': 0,
                    'throughput': 0
                })
    
    total_test_time = time.time() - start_time
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    avg_time = sum(r['total_time'] for r in results) / len(results) if results else 0
    total_bytes = sum(r['bytes_transferred'] for r in results)
    avg_throughput = total_bytes / total_test_time if total_test_time > 0 else 0
    
    return {
        'total_time': total_test_time,
        'avg_time_per_client': avg_time,
        'throughput': avg_throughput,
        'successful_workers': successful,
        'failed_workers': failed,
        'results': results
    }

def run_multiprocessing_stress_test(operation: str, host: str, port: int, filename: str, 
                                  file_data: bytes, num_workers: int, timeout: int = 60) -> Dict:
    """Run stress test using ProcessPoolExecutor"""
    
    if operation == 'upload':
        worker_args = [(i, host, port, filename, file_data, timeout) for i in range(num_workers)]
        worker_func = worker_upload_test
    else:  # download
        worker_args = [(i, host, port, filename, timeout) for i in range(num_workers)]
        worker_func = worker_download_test
    
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_func, args) for args in worker_args]
        
        for future in futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                results.append({
                    'worker_id': -1,
                    'success': False,
                    'message': str(e),
                    'total_time': 0,
                    'request_time': 0,
                    'bytes_transferred': 0,
                    'throughput': 0
                })
    
    total_test_time = time.time() - start_time
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    avg_time = sum(r['total_time'] for r in results) / len(results) if results else 0
    total_bytes = sum(r['bytes_transferred'] for r in results)
    avg_throughput = total_bytes / total_test_time if total_test_time > 0 else 0
    
    return {
        'total_time': total_test_time,
        'avg_time_per_client': avg_time,
        'throughput': avg_throughput,
        'successful_workers': successful,
        'failed_workers': failed,
        'results': results
    }

if __name__ == "__main__":
    # Simple CLI for testing
    client = FileClient()
    
    while True:
        command = input("Enter command (LIST, UPLOAD <file>, DOWNLOAD <file>, QUIT): ").strip()
        
        if command == "QUIT":
            break
        elif command == "LIST":
            success, files, time_taken = client.list_files()
            print(f"Files: {files} (took {time_taken:.2f}s)")
        elif command.startswith("UPLOAD"):
            parts = command.split()
            if len(parts) == 2:
                filename = parts[1]
                success, message, time_taken, bytes_transferred = client.upload_file(filename)
                print(f"Upload: {message} ({bytes_transferred} bytes in {time_taken:.2f}s)")
        elif command.startswith("DOWNLOAD"):
            parts = command.split()
            if len(parts) == 2:
                filename = parts[1]
                success, message, time_taken, bytes_transferred = client.download_file(filename, f"downloaded_{filename}")
                print(f"Download: {message} (took {time_taken:.2f}s)")