from socket import socket, AF_INET, SOCK_STREAM
import base64

def send_request(user_command):
    host_info = ('localhost', 45000)
    tcp_socket = socket(AF_INET, SOCK_STREAM)
    tcp_socket.connect(host_info)

    try:
        tcp_socket.sendall(f"{user_command}\r\n".encode('utf-8'))
        server_reply = tcp_socket.recv(4096)
        print("Received:", server_reply.decode('utf-8'))
    except Exception as error:
        print(f"Error: {error}")
    finally:
        tcp_socket.close()

def main():
    while True:
        user_input = input("Enter command (LIST, GET, IMAGE, UPLOAD, DELETE, QUIT): ").strip()
        if user_input == "LIST" or user_input == "IMAGE" or user_input.startswith("GET") or user_input.startswith("DELETE"):
            send_request(user_input)
        elif user_input.startswith("UPLOAD"):
            target_file = input("Enter the filename to upload: ").strip()
            with open(target_file, 'rb') as file_handle:
                encoded_content = base64.b64encode(file_handle.read()).decode('utf-8')
            send_request(f"UPLOAD {target_file} {encoded_content}")
        elif user_input == "QUIT":
            break
        else:
            print("Invalid command.")

if __name__ == "__main__":
    main()