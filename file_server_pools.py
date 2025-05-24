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
        self.server_socket.listen(10)
        
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

def handle_client_connection_process(client_data):
    """Handle client connection for multiprocessing"""
    client_conn, client_addr = client_data
    protocol_handler = FileProtocol()
    return handle_client_connection(client_conn, client_addr, protocol_handler)

class ProcessPoolServer:
    def __init__(self, server_ip='0.0.0.0', server_port=45001, max_workers=5):
        self.server_address = (server_ip, server_port)
        self.max_workers = max_workers
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.running = True
        self.successful_connections = 0
        self.failed_connections = 0

    def start(self):
        logging.info(f"ProcessPool Server starting on {self.server_address} with {self.max_workers} workers")
        self.server_socket.bind(self.server_address)
        self.server_socket.listen(10)
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            while self.running:
                try:
                    client_conn, client_addr = self.server_socket.accept()
                    logging.info(f"Connection from {client_addr}")
                    
                    # For multiprocessing, we need to handle socket differently
                    # This is a simplified approach - in practice, you might want to use queues
                    future = executor.submit(handle_client_connection_process, (client_conn, client_addr))
                    futures.append(future)
                    
                    # Clean up completed futures
                    futures = [f for f in futures if not f.done()]
                    
                except Exception as e:
                    if self.running:
                        logging.error(f"Error accepting connection: {e}")
    
    def stop(self):
        self.running = False
        self.server_socket.close()

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
    if len(sys.argv) > 1:
        if sys.argv[1] == "thread":
            workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            start_thread_pool_server(workers=workers)
        elif sys.argv[1] == "process":
            workers = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            start_process_pool_server(workers=workers)
    else:
        print("Usage: python file_server_pools.py [thread|process] [num_workers]")