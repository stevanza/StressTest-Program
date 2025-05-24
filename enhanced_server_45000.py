#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run file server with enhanced settings
from fixed_file_server_pools import main
import threading
import socket

# Override default settings for high-load scenarios
original_socket = socket.socket

def enhanced_socket(*args, **kwargs):
    s = original_socket(*args, **kwargs)
    # Set socket options for better Windows performance
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        # Increase socket buffer sizes
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
    except OSError:
        pass
    return s

# Monkey patch socket creation
socket.socket = enhanced_socket

# Set environment variables for server configuration
os.environ['SERVER_PORT'] = '45000'
os.environ['SERVER_BACKLOG'] = '100'
os.environ['SERVER_TIMEOUT'] = '30'

if __name__ == "__main__":
    sys.argv = ['file_server_pools.py', 'thread', '1']
    main()
