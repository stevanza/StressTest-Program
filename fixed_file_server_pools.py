import os
import json
import base64
import logging
import socket
import threading
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from glob import glob
import shlex
from functools import partial
import pickle
import struct

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FileInterface:
    def __init__(self):
        if not os.path.exists('files/'):
            os.makedirs('files/')
        self.base_path = os.path.abspath('files/')

    def list(self, arguments=[]):
        try:
            os.chdir(self.base_path)
            available_files = glob('*.*')
            return dict(status='OK', data=available_files)
        except Exception as error:
            return dict(status='ERROR', data=str(error))

    def get(self, arguments=[]):
        try:
            if not arguments or arguments[0] == '':
                return dict(status='ERROR', data='Filename not provided')
            
            target_filename = arguments[0]
            file_path = os.path.join(self.base_path, target_filename)
            
            if not os.path.exists(file_path):
                return dict(status='ERROR', data='File not found')
                
            with open(file_path, 'rb') as file_pointer:
                encoded_file_data = base64.b64encode(file_pointer.read()).decode()
            return dict(status='OK', data_namafile=target_filename, data_file=encoded_file_data)
        except Exception as error:
            return dict(status='ERROR', data=str(error))

    def upload(self, arguments=[]):
        try:
            if len(arguments) < 2:
                return dict(status='ERROR', data='Insufficient arguments')
                
            new_filename = arguments[0]
            file_data = arguments[1]
            
            # Ensure correct padding
            padding_needed = len(file_data) % 4
            if padding_needed != 0:
                file_data += '=' * (4 - padding_needed)
                
            decoded_content = base64.b64decode(file_data)
            file_path = os.path.join(self.base_path, new_filename)
            
            with open(file_path, 'wb') as file_writer:
                file_writer.write(decoded_content)
            return dict(status='OK', data=f"{new_filename} uploaded successfully")
        except Exception as error:
            return dict(status='ERROR', data=str(error))

    def delete(self, arguments=[]):
        try:
            if not arguments:
                return dict(status='ERROR', data='Filename not provided')
                
            file_to_remove = arguments[0]
            file_path = os.path.join(self.base_path, file_to_remove)
            
            if not os.path.exists(file_path):
                return dict(status='ERROR', data='File not found')
                
            os.remove(file_path)
            return dict(status='OK', data=f"{file_to_remove} deleted successfully")
        except Exception as error:
            return dict(status='ERROR', data=str(error))

    def image(self, arguments=[]):
        try:
            os.chdir(self.base_path)
            image_files = glob('*.jpg') + glob('*.png') + glob('*.jpeg')
            return dict(status='OK', data=image_files)
        except Exception as error:
            return dict(status='ERROR', data=str(error))

class FileProtocol:
    def __init__(self):
        self.file_manager = FileInterface()

    def process_string(self, incoming_data=''):
        try:
            parsed_command = shlex.split(incoming_data.strip())
            if not parsed_command:
                return json.dumps(dict(status='ERROR', data='Empty command'))
                
            request_type = parsed_command[0].strip().lower()
            command_params = parsed_command[1:]
            
            if hasattr(self.file_manager, request_type):
                operation_result = getattr(self.file_manager, request_type)(command_params)
                return json.dumps(operation_result)
            else:
                return json.dumps(dict(status='ERROR', data='Unknown command'))
        except Exception as error:
            logging.error(f"Error processing command: {error}")
            return json.dumps(dict(status='ERROR', data=str(error)))

def handle_client_connection(client_conn, client_addr, protocol_handler):
    """Handle individual client connection"""
    success = True
    try:
        while True:
            received_data = client_conn.recv(8192)
            if not received_data:
                break
                
            decoded_message = received_data.decode().strip()
            if decoded_message:
                processed_result = protocol_handler.process_string(decoded_message)
                response_data = processed_result + "\r\n\r\n"
                client_conn.sendall(response_data.encode())
    except Exception as e:
        logging.error(f"Error handling client {client_addr}: {e}")
        success = False
    finally:
        client_conn.close()
    
    return success

class ThreadPoolServer:
    def __init__(self, server_ip='0.0.0.0', server_port=45000, max_workers=5):
        # Check for SERVER_PORT environment variable
        port_from_env = os.environ.get('SERVER_PORT')
        if port_from_env:
            try:
                server_port = int(port_from_env)
                logging.info(f"Using port from environment: {server_port}")
            except ValueError:
                logging.warning(f"Invalid SERVER_PORT value: {port_from_env}, using default {server_port}")
        
        self.server_address = (server_ip, server_port)
        self.max_workers = max_workers
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.protocol_handler = FileProtocol()
        self.running = True
        self.successful_connections = 0
        self.failed_connections = 0
        self.lock = threading.Lock()

    def start(self):
        logging.info(f"ThreadPool Server starting on {self.server_address} with {self.max_workers} workers")
        self.server_socket.bind(self.server_address)
        self.server_socket.listen(100)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while self.running:
                try:
                    client_conn, client_addr = self.server_socket.accept()
                    logging.info(f"Connection from {client_addr}")
                    
                    future = executor.submit(handle_client_connection, client_conn, client_addr, self.protocol_handler)
                    
                    # Optional: track completion
                    def connection_completed(fut):
                        with self.lock:
                            if fut.result():
                                self.successful_connections += 1
                            else:
                                self.failed_connections += 1
                    
                    future.add_done_callback(connection_completed)
                    
                except Exception as e:
                    if self.running:
                        logging.error(f"Error accepting connection: {e}")
    
    def stop(self):
        self.running = False
        self.server_socket.close()
    
    def get_stats(self):
        return self.successful_connections, self.failed_connections

# Process Pool Server Implementation
def process_worker(worker_id, task_queue, result_queue):
    """Worker process that handles client requests"""
    protocol_handler = FileProtocol()
    logging.info(f"Worker {worker_id} started")
    
    while True:
        try:
            # Get task from queue
            task = task_queue.get(timeout=1)
            
            if task is None:  # Poison pill
                logging.info(f"Worker {worker_id} shutting down")
                break
                
            client_data, client_addr = task
            
            # Process the client data
            try:
                decoded_message = client_data.decode().strip()
                if decoded_message:
                    processed_result = protocol_handler.process_string(decoded_message)
                    result_queue.put((client_addr, processed_result))
                else:
                    result_queue.put((client_addr, None))
            except Exception as e:
                logging.error(f"Worker {worker_id} error processing request: {e}")
                error_result = json.dumps(dict(status='ERROR', data=str(e)))
                result_queue.put((client_addr, error_result))
                
        except:
            # Timeout or other error, continue
            continue
    
    logging.info(f"Worker {worker_id} stopped")

class ProcessPoolServer:
    def __init__(self, server_ip='0.0.0.0', server_port=45001, max_workers=5):
        # Check for SERVER_PORT environment variable
        port_from_env = os.environ.get('SERVER_PORT')
        if port_from_env:
            try:
                server_port = int(port_from_env)
                logging.info(f"Using port from environment: {server_port}")
            except ValueError:
                logging.warning(f"Invalid SERVER_PORT value: {port_from_env}, using default {server_port}")
        
        self.server_address = (server_ip, server_port)
        self.max_workers = max_workers
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.running = True
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.workers = []
        self.client_connections = {}
        self.connection_lock = threading.Lock()

    def start(self):
        logging.info(f"ProcessPool Server starting on {self.server_address} with {self.max_workers} workers")
        self.server_socket.bind(self.server_address)
        self.server_socket.listen(100)
        
        # Start worker processes
        for i in range(self.max_workers):
            worker = multiprocessing.Process(
                target=process_worker,
                args=(i, self.task_queue, self.result_queue)
            )
            worker.start()
            self.workers.append(worker)
        
        # Start result handler thread
        result_handler = threading.Thread(target=self.handle_results)
        result_handler.daemon = True
        result_handler.start()
        
        # Accept connections in main thread
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                try:
                    client_conn, client_addr = self.server_socket.accept()
                    logging.info(f"Connection from {client_addr}")
                    
                    # Handle connection in a separate thread
                    handler_thread = threading.Thread(
                        target=self.handle_client_connection_thread,
                        args=(client_conn, client_addr)
                    )
                    handler_thread.daemon = True
                    handler_thread.start()
                    
                except socket.timeout:
                    continue
                    
            except Exception as e:
                if self.running:
                    logging.error(f"Error accepting connection: {e}")
    
    def handle_client_connection_thread(self, client_conn, client_addr):
        """Handle client connection in a separate thread"""
        try:
            while True:
                received_data = client_conn.recv(8192)
                if not received_data:
                    break
                
                # Store connection for later response
                with self.connection_lock:
                    self.client_connections[client_addr] = client_conn
                
                # Send data to worker queue
                self.task_queue.put((received_data, client_addr))
                
        except Exception as e:
            logging.error(f"Error handling client {client_addr}: {e}")
        finally:
            # Clean up connection
            with self.connection_lock:
                if client_addr in self.client_connections:
                    del self.client_connections[client_addr]
            client_conn.close()
    
    def handle_results(self):
        """Handle results from worker processes"""
        while self.running:
            try:
                # Get result from queue
                client_addr, result = self.result_queue.get(timeout=1)
                
                # Send response back to client
                with self.connection_lock:
                    if client_addr in self.client_connections:
                        client_conn = self.client_connections[client_addr]
                        if result:
                            response_data = result + "\r\n\r\n"
                            client_conn.sendall(response_data.encode())
                        
            except:
                # Timeout or other error, continue
                continue
    
    def stop(self):
        self.running = False
        
        # Stop workers
        for _ in self.workers:
            self.task_queue.put(None)  # Poison pill
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join()
        
        # Close server socket
        try:
            self.server_socket.close()
        except:
            pass
        
        logging.info("ProcessPool Server stopped")

def start_thread_pool_server(port=45000, workers=5):
    """Start thread pool server"""
    server = ThreadPoolServer(server_port=port, max_workers=workers)
    try:
        server.start()
    except KeyboardInterrupt:
        logging.info("Shutting down thread pool server")
    finally:
        server.stop()
    return server.get_stats()

def start_process_pool_server(port=45001, workers=5):
    """Start process pool server"""
    server = ProcessPoolServer(server_port=port, max_workers=workers)
    try:
        server.start()
    except KeyboardInterrupt:
        logging.info("Shutting down process pool server")
    finally:
        server.stop()

if __name__ == "__main__":
    import sys
    
    # Set multiprocessing start method for Windows compatibility
    if sys.platform == 'win32':
        multiprocessing.set_start_method('spawn', force=True)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "thread":
            workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            start_thread_pool_server(workers=workers)
        elif sys.argv[1] == "process":
            workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            start_process_pool_server(workers=workers)
    else:
        print("Usage: python file_server_pools.py [thread|process] [num_workers]")