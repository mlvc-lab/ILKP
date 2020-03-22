import pathlib
import argparse
import requests
from tqdm import tqdm


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 1024 * 32

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE), desc=destination, miniters=0, unit='MB', unit_scale=1/32, unit_divisor=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download ImageNet')
    parser.add_argument('--datapath', default='../data', type=str, metavar='PATH',
                        help='Where you want to save ImageNet? (default: ../data)')
    args = parser.parse_args()

    data_dir = pathlib.Path(args.datapath)
    data_dir.mkdir(parents=True, exist_ok=True)
    files = [
        ('1gYyRk6cAxkdhKaRbm2etrj-L6yWbaIxR', 'ILSVRC2012_devkit_t12.tar.gz'),
        ('1nrD10h3H4abgZr4ToNvSAesMR9CQD4Dk', 'ILSVRC2012_img_train.tar'),
        ('1BNzST-vesjJz4_JEYIsfGvDFXv_0qBNu', 'ILSVRC2012_img_val.tar'),
        ('14pb7YLYBRyp4QBDu4qF6SOfZDogXza01', 'meta.bin')
    ]

    for file_id, file_name in files:
        destination = data_dir / file_name
        download_file_from_google_drive(file_id, destination.as_posix())
