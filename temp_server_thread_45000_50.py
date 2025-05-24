
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fixed_file_server_pools import ThreadPoolServer, ProcessPoolServer
import logging
import signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def signal_handler(sig, frame):
    logging.info("Server shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    if "thread" == "thread":
        server = ThreadPoolServer(server_port=45000, max_workers=50)
    else:
        server = ProcessPoolServer(server_port=45000, max_workers=50)
    
    logging.info(f"Starting {server_type} server on port 45000 with 50 workers")
    
    try:
        server.start()
    except KeyboardInterrupt:
        logging.info("Server interrupted")
    except Exception as e:
        logging.error(f"Server error: {e}")
    finally:
        server.stop()
        logging.info("Server stopped")
