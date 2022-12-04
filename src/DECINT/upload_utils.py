import requests
import shutil
import os
import json
import socket

def send(host, message, port=1379):
    """
    sends a message to the given host
    tries the default port and if it doesn't work search for actual port
    this process is skipped if send to all for speed
    """
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect((host, port))
        client.sendall(message.encode("utf-8"))
        print(f"Message to {host} {message}\n")
    except ConnectionError:
        pass # TODO add error handling

    client.close()

def zip_folder(folder_path):
    shutil.make_archive(f"{os.path.dirname(__file__)}/temp", 'zip', folder_path)

def upload_script_to_nodes(source_path, nodes):
    #nodes = requests.get("http://DECINT/get_nodes").json()
    zip_folder(source_path)
    with open(f"{os.path.dirname(__file__)}/temp.zip", "rb") as file:
        data = file.read()
    for node in nodes:
        send(node["ip"], f"UPLOAD {data}", node["port"])

def uplaod_data_to_node(data_path, nodes):
    zip_folder(data_path)
    with open(f"{os.path.dirname(__file__)}/temp.zip", "rb") as file:
        data = file.read()
    for node in nodes:
        send(node["ip"], f"DATA {data}", node["port"])

