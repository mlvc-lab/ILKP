import pathlib
import argparse
import requests
from tqdm import tqdm

import models as models
from data import valid_datasets as dataset_names


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


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
    parser = argparse.ArgumentParser(description='PyTorch TestBench for Weight Prediction')
    parser.add_argument('dataset', metavar='DATA', default='cifar10',
                        choices=dataset_names,
                        help='dataset: ' +
                             ' | '.join(dataset_names) +
                             ' (default: cifar10)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='mobilenet',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: mobilenet)')
    parser.add_argument('--layers', default=16, type=int, metavar='N',
                        help='number of layers in VGG/ResNet/ResNeXt/WideResNet (default: 16)')
    parser.add_argument('-o', '--out', type=str, default='pretrained_model.pth',
                        help='output filename of pretrained model from our google drive')
    args = parser.parse_args()

    # 'file_id' is 
    if args.dataset == 'cifar10':
        if args.arch == 'mobilenet':
            file_id = '1vFf4ipUo1ZvQSo17AX_jFMyS3ZVX79lb'
        elif args.arch == 'mobilenetv2':
            file_id = '19qGclWiEdBz40af3O3vd0S1cTq_oNcLM'
        elif args.arch == 'shufflenet':
            file_id = '1qG9Dt_GRxecYMCkthXd9_dAJ6Dv6EXop'
        elif args.arch == 'shufflenetv2':
            file_id = '1VdlJSXt4PWi6iWZNYDNFZaUaIZgtRavg'
        elif args.arch == 'resnet':
            if args.layers == 20:
                file_id = '1h4f61kCjK25rhIfwZf6uj-NmPnRrfoBy'
            elif args.layers == 32:
                file_id = '1Zhljp1SRq1EHGds86EzMz_SH0XUDVmsF'
            elif args.layers == 44:
                file_id = '1H2WnlFQq-m6sQGY7Gj_E0SmXr1FhqqRY'
            elif args.layers == 56:
                file_id = '1B-_ixX6QtxilPUiX2UCzCNNK1Dt3zZR-'
            elif args.layers == 110:
                file_id = '1BYItn3cyeSbaUSB3RwzoEVzdlVE5rczx'
        else:
            print('Not prepared yet..\nProgram exit...')
            exit()
    elif args.dataset == 'cifar100':
        if args.arch == 'mobilenet':
            file_id = '1F3lztkxgQA5TXFB5-MbW5D3FGA5utC2a'
        elif args.arch == 'mobilenetv2':
            file_id = '1qKI8Ipjs33eBGBWoy4L202Bmuzn4X9TC'
        elif args.arch == 'shufflenet':
            file_id = '1TrV7OrUNJ0eIgJNR1tR5JupX8sRhFkZK'
        elif args.arch == 'shufflenetv2':
            file_id = '1VcRXdeGYPtwz75Qx_NF5SyXI-yRM3Ivf'
        elif args.arch == 'resnet':
            if args.layers == 20:
                file_id = '1DtEsqiN5UzMn07nQcGmYfu1cBzpnERm-'
            elif args.layers == 32:
                file_id = '1xmUR5iCIUdWeHH2Yv9NWg6fNokRIQDE8'
            elif args.layers == 44:
                file_id = '1kKKSzWceAHUruwfJbOdMe2QM7eLbsv5p'
            elif args.layers == 56:
                file_id = '1TpUSzrPfRWvf6ClbKnfXHMQfYcFWTqrX'
            elif args.layers == 110:
                file_id = '13WaEbA0y1aKs2aWHS8hgnIlePpg6250f'
        else:
            print('Not prepared yet..\nProgram exit...')
            exit()
    elif args.dataset == 'imagenet':
        if args.arch == 'mobilenet':
            file_id = '1qzyIMwH-Nd8wM77ScJmY7CXUhJYzYG2i'
        elif args.arch == 'mobilenetv2':
            file_id = '15FpbDFnVPJSNfljG6DC2HzhgS_Oif1Ab'
        # elif args.arch == 'shufflenet':
        #     file_id = ''
        # elif args.arch == 'shufflenetv2':
        #     file_id = ''
        # elif args.arch == 'resnet':
        #     if args.layers == 18:
        #         file_id = ''
        #     if args.layers == 34:
        #         file_id = ''
        #     if args.layers == 50:
        #         file_id = ''
        #     if args.layers == 101:
        #         file_id = ''
        #     if args.layers == 152:
        #         file_id = ''
        else:
            print('Not prepared yet..\nProgram exit...')
            exit()
    else:
        print('Not prepared yet..\nProgram exit...')
        exit()
    
    hasDiffLayersArchs = ['vgg', 'resnet', 'resnext', 'wideresnet']

    ckpt_dir = pathlib.Path('checkpoint')
    if args.arch in hasDiffLayersArchs:
        dir_path = ckpt_dir / (args.arch + str(args.layers)) / args.dataset
    else:
        dir_path = ckpt_dir / args.arch / args.dataset
    dir_path.mkdir(parents=True, exist_ok=True)
    destination = dir_path / args.out

    download_file_from_google_drive(file_id, destination.as_posix())
