import requests
from PIL import Image

DOWNLOAD_DIR = "..\\..\\static\\img"


def download_image(name: str, url: str) -> None:
    """ Get persons from page with type and link to interview
    :param name: name of person from the image
    :param url: url of the image
    """
    data = requests.get(url).content
    with open(f'{DOWNLOAD_DIR}\\{name}.jpg', 'wb') as f:
        f.write(data)
