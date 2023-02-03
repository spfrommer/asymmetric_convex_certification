import click

import torch
import numpy as np
import itertools

from convexrobust.data import datamodules
from convexrobust.utils import file_utils, dirs
from convexrobust.utils import torch_utils as TU
from convexrobust.model.certificate import Certificate, Norm

@click.command()
@click.option('--data', type=click.Choice(datamodules.names), default='mnist_38')
@click.option('--model_name', type=str, default='abcrown')
def run(data, model_name):
    experiment_directory = f'{data}-standard'
    root_path = dirs.out_path(experiment_directory, 'abcrown_results')
    results_path = dirs.out_path(experiment_directory, 'results.pkl')
    data_path = dirs.out_path(experiment_directory, 'data.pkl')

    results = file_utils.read_pickle(results_path)
    data = file_utils.read_pickle(data_path)
    signals, targets = torch.cat(data['signals']), torch.cat(data['targets'])

    for i, (signal, target) in enumerate(zip(signals, targets)):
        if target == TU.CERT_CLASS:
            # Zero out existing certificates
            results[model_name][i].certificate = Certificate.zero()

    files = file_utils.files_with_extension(root_path, 'npy')
    for f in files:
        summary_data = np.load(dirs.path(root_path, f), allow_pickle=True)['summary']
        cert_indices = get_cert_indices(summary_data)

        eps, norm = f[:-4].split('_')
        eps, norm = float(eps), Norm(float(norm))

        cert_class_counter = 0
        for i, (signal, target) in enumerate(zip(signals, targets)):
            if target == TU.CERT_CLASS:
                certificate = results[model_name][i].certificate
                if certificate is None:
                    certificate = Certificate.zero()
                if cert_class_counter in cert_indices:
                    certificate.radius[norm] = max(eps, certificate.radius[norm])
                results[model_name][i].certificate = certificate
                cert_class_counter += 1

    file_utils.write_pickle(results_path, results)

def get_cert_indices(summary_data):
    safe_keys = [k for k in summary_data.keys() if 'safe' in k and 'unsafe' not in k]
    safe_values = [summary_data[k] for k in safe_keys]
    return list(itertools.chain(*safe_values))



if __name__ == "__main__":
    run()
