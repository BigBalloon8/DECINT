import click
import requests
import upload_utils


@click.command()
@click.option("--script_dir", "-sd", help="Directory of the script to be uploaded")
@click.option("--data_dir", "-dd", help="Directory of the data to be uploaded")
def run(script_dir, data_dir):
    nodes = requests.get("http://DECINT/get_nodes").json()
    upload_utils.upload_script_to_nodes(script_dir, nodes)
    upload_utils.uplaod_data_to_node(data_dir, nodes)


