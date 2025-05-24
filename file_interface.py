import os
import json
import base64
from glob import glob

class FileInterface:
    def __init__(self):
        if not os.path.exists('files/'):
            os.makedirs('files/')
        os.chdir('files/')

    def list(self, arguments=[]):
        try:
            available_files = glob('*.*')
            return dict(status='OK', data=available_files)
        except Exception as error:
            return dict(status='ERROR', data=str(error))

    def get(self, arguments=[]):
        try:
            target_filename = arguments[0]
            if target_filename == '':
                return None
            with open(f"{target_filename}", 'rb') as file_pointer:
                encoded_file_data = base64.b64encode(file_pointer.read()).decode()
            return dict(status='OK', data_namafile=target_filename, data_file=encoded_file_data)
        except Exception as error:
            return dict(status='ERROR', data=str(error))

    def upload(self, arguments=[]):
        try:
            new_filename = arguments[0]
            file_data = arguments[1]
            # Ensure correct padding
            padding_needed = len(file_data) % 4
            if padding_needed != 0:
                file_data += '=' * (4 - padding_needed)
            decoded_content = base64.b64decode(file_data)
            with open(new_filename, 'wb') as file_writer:
                file_writer.write(decoded_content)
            return dict(status='OK', data=f"{new_filename} uploaded successfully")
        except Exception as error:
            return dict(status='ERROR', data=str(error))

    def delete(self, arguments=[]):
        try:
            file_to_remove = arguments[0]
            os.remove(file_to_remove)
            return dict(status='OK', data=f"{file_to_remove} deleted successfully")
        except Exception as error:
            return dict(status='ERROR', data=str(error))

    def image(self, arguments=[]):
        try:
            image_files = glob('*.jpg') + glob('*.png') + glob('*.jpeg')
            return dict(status='OK', data=image_files)
        except Exception as error:
            return dict(status='ERROR', data=str(error))

if __name__ == '__main__':
    file_handler = FileInterface()
    print(file_handler.list())
    print(file_handler.get(['example.jpg']))
    print(file_handler.upload(['example_upload.jpg', base64.b64encode(b'This is a test file content').decode('utf-8')]))
    print(file_handler.delete(['example_upload.jpg']))
    print(file_handler.image())