import os

import torch
from torch import Tensor
from pytorch_lightning import LightningDataModule

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import foolbox

from convexrobust.model.certificate import Certificate, Norm
from convexrobust.model.base_certifiable import BaseCertifiable
from convexrobust.model.randsmooth_certifiable import RandsmoothCertifiable
from convexrobust.utils import dirs, file_utils, pretty
from convexrobust.utils import torch_utils as TU
from convexrobust.main.train import ModelDict, BlueprintDict


from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Result:
    target: Tensor
    pred: Tensor
    certificate: Optional[Certificate]  # Only if target is class 0
    empirical_certificate: Optional[Certificate]

@dataclass
class EvaluateConfig:
    """Loaded from the CLI arguments."""
    verify_cert: bool = False
    empirical_cert: bool = False
    # No eval_n as this is loaded into the datamodule eval_iterator


ResultDict = Dict[str, List[Result]]


def evaluate_models(
        models: ModelDict, blueprints: BlueprintDict,
        experiment_directory: str, datamodule: LightningDataModule, config: EvaluateConfig
    ) -> ResultDict:
    results_path = dirs.out_path(f'{experiment_directory}', 'results.pkl')
    data_path = dirs.out_path(f'{experiment_directory}', 'data.pkl')

    for (_, model) in models.items():
        model.eval()

    results = init_eval_results(models, blueprints, results_path)

    models_to_eval = { n: m for (n, m) in models.items() if not blueprints[n].load_eval_results }

    pretty.subsection_print(f'Running evaluation...')
    signals, targets = [], []
    for (signal, target) in datamodule.eval_iterator(do_tqdm=True):
        for (name, model) in models_to_eval.items():
            if target == TU.CERT_CLASS and not model.external_certification:
                pred, certificate = model.certify(signal, target)
                if pred != target:
                    certificate = certificate.zero()
            else:
                pred, certificate = model.predict(signal), None

            assert pred.shape == torch.Size([1])

            if config.verify_cert and certificate is not None:
                verify_radii(model, certificate, signal, target)

            emp_certificate = None
            if config.empirical_cert:
                emp_certificate = empirical_certificate(model, signal, target)

            results[name].append(Result(target, pred, certificate, emp_certificate))

        signals.append(signal)
        targets.append(target)

    pretty.subsection_print('Writing results')
    file_utils.write_pickle(results_path, results)
    file_utils.write_pickle(data_path, {'signals': signals, 'targets': targets})
    return results


def init_eval_results(
        models: ModelDict, blueprints: BlueprintDict, results_path: str
    ) -> ResultDict:

    results: ResultDict = {}

    if os.path.isfile(results_path):
        pretty.subsection_print('Found previous results...')
        results = file_utils.read_pickle(results_path)

    # Remove old results for models not being evaluated
    # results = {k: results[k] for k in models.keys() if k in results.keys()}

    for (name, model) in models.items():
        if not (name in results and blueprints[name].load_eval_results):
            # Clear eval results that should not be loaded
            results[name] = []
        assert not model.training
        if blueprints[name].load_eval_results:
            print(f'Loading old eval results for {name}')
        else:
            print(f'Evaluating {name} with balance: {model.class_balance.item():0.3f}')

    return results


def verify_radii(
        model: BaseCertifiable, certificate: Certificate, signal: Tensor, target: Tensor
    ) -> None:

    if isinstance(model, RandsmoothCertifiable):
        return # Don't verify nondeterministic certificates

    fb_model = foolbox.models.PyTorchModel(model, bounds=(0, 1))
    attacks = {
        Norm.L1: foolbox.attacks.L1ProjectedGradientDescentAttack(steps=100),
        Norm.L2: foolbox.attacks.L2ProjectedGradientDescentAttack(steps=100),
        Norm.LInf: foolbox.attacks.LinfProjectedGradientDescentAttack(steps=300)
    }

    for norm in [Norm.L1, Norm.L2, Norm.LInf]:
        if certificate.radius[norm] > 0.0:
            attack = attacks[norm]
            advs, advs_clipped, is_adv = attack(
                fb_model, signal, target, epsilons=[certificate.radius[norm]]
            )
            if is_adv:
                print('Certification failed...')
                import pdb; pdb.set_trace()


def empirical_certificate(model: BaseCertifiable, signal: Tensor, target: Tensor) -> Certificate:
    fb_model = foolbox.models.PyTorchModel(model, bounds=(0, 1))
    attack = foolbox.attacks.LInfFMNAttack()
    advs, _, _ = attack(fb_model, signal, target, epsilons=None)
    empirical_certificate = Certificate({
        Norm.L1: (advs - signal).norm(1).item(),
        Norm.L2: (advs - signal).norm(2).item(),
        Norm.LInf: (advs - signal).norm(float('inf')).item()
    })
    return empirical_certificate
