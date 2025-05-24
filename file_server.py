from socket import *
import socket
import threading
import logging
import sys
from file_protocol import FileProtocol

# Inisialisasi FileProtocol
protocol_handler = FileProtocol()

class ProcessTheClient(threading.Thread):
    def __init__(self, client_conn, client_addr):
        self.client_conn = client_conn
        self.client_addr = client_addr
        threading.Thread.__init__(self)

    def run(self):
        while True:
            received_data = self.client_conn.recv(4096)  # Meningkatkan ukuran buffer untuk menerima lebih banyak data
            if received_data:
                decoded_message = received_data.decode()
                processed_result = protocol_handler.proses_string(decoded_message)
                response_data = processed_result + "\r\n\r\n"
                self.client_conn.sendall(response_data.encode())
            else:
                break
        self.client_conn.close()

class Server(threading.Thread):
    def __init__(self, server_ip='0.0.0.0', server_port=45000):  # Mengubah port sesuai instruksi
        self.server_address = (server_ip, server_port)
        self.active_clients = []
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        threading.Thread.__init__(self)

    def run(self):
        logging.warning(f"Server berjalan di ip address {self.server_address}")
        self.server_socket.bind(self.server_address)
        self.server_socket.listen(1)
        while True:
            self.client_conn, self.remote_address = self.server_socket.accept()
            logging.warning(f"Connection from {self.remote_address}")

            client_thread = ProcessTheClient(self.client_conn, self.remote_address)
            client_thread.start()
            self.active_clients.append(client_thread)

def main():
    file_server = Server(server_ip='0.0.0.0', server_port=45000)
    file_server.start()

if __name__ == "__main__":
    main()