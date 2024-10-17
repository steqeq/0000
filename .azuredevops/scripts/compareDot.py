import os
import requests
import hashlib
import sys

def download_dotfile(url, file_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download file from {url}. Status code: {response.status_code}")

def hash_file(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def compare_files(local_file, downloaded_file):
    local_hash = hash_file(local_file)
    downloaded_hash = hash_file(downloaded_file)
    return local_hash == downloaded_hash

def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    local_dotfile = os.path.join(script_directory, 'input.dot')
    downloaded_dotfile = os.path.join(script_directory, 'dependency_graph.dot')

    url = 'https://github.com/ROCm/ROCm/blob/generatedependencygraph/.azuredevops/scripts/dependency_graph.dot'

    try:
        download_dotfile(url, downloaded_dotfile)
        if compare_files(local_dotfile, downloaded_dotfile):
            print("The local DOT file and the downloaded DOT file are the same.")
        else:
            print("The local DOT file and the downloaded DOT file are different.")
            # Exit with a non-zero status to signal failed/unstable build on Jenkins
            # to trigger post-build email
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
