import json
import logging
import shlex
from file_interface import FileInterface

class FileProtocol:
    def __init__(self):
        self.file_manager = FileInterface()

    def proses_string(self, incoming_data=''):
        logging.warning(f"string diproses: {incoming_data}")
        try:
            parsed_command = shlex.split(incoming_data)
            request_type = parsed_command[0].strip().lower()  # Mengubah perintah menjadi huruf kecil
            logging.warning(f"memproses request: {request_type}")
            command_params = parsed_command[1:]
            logging.warning(f"params: {command_params}")
            operation_result = getattr(self.file_manager, request_type)(command_params)
            return json.dumps(operation_result)
        except Exception as error:
            logging.error(f"Error: {error}")
            return json.dumps(dict(status='ERROR', data=str(error)))