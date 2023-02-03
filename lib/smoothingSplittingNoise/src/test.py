import numpy as np
import pathlib
import os
import sys
import torch
import torch.nn as nn
from argparse import ArgumentParser
from torchnet import meter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from src.models import *
from src.attacks import *
from src.smooth import *
from src.noises import *
from src.datasets import *
import time
from src.lib.dncnn import DnCNN
if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--device", default="cuda", type=str)
    argparser.add_argument("--batch-size", default=2, type=int)
    argparser.add_argument("--num-workers", default=min(os.cpu_count(), 8), type=int)
    argparser.add_argument("--sample-size-pred", default=64, type=int)
    argparser.add_argument("--sample-size-cert", default=100000, type=int)
    argparser.add_argument("--noise-batch-size", default=512, type=int)
    argparser.add_argument("--sigma", default=0.0, type=float)
    argparser.add_argument("--noise", default="Clean", type=str)
    argparser.add_argument("--k", default=None, type=int)
    argparser.add_argument("--j", default=None, type=int)
    argparser.add_argument("--a", default=None, type=int)
    argparser.add_argument("--lambd", default=None, type=float)
    argparser.add_argument("--dataset-skip", default=1, type=int)
    argparser.add_argument("--experiment-name", default="cifar", type=str)
    argparser.add_argument("--dataset", default="cifar", type=str)
    argparser.add_argument("--model", default="WideResNet", type=str)
    argparser.add_argument("--rotate", action="store_true")
    argparser.add_argument("--output-dir", type=str, default=os.getenv("PT_OUTPUT_DIR"))
    argparser.add_argument("--save-path", type=str, default=None)
    argparser.add_argument("--denoiser-path", type=str, default=None)
    argparser.add_argument("--with_denoiser", action="store_true")

    args = argparser.parse_args()
    args.seed = None # Test for derandomized done elsewhere
    test_dataset = get_dataset(args.dataset, "test")
    test_dataset = Subset(test_dataset, list(range(0, len(test_dataset), args.dataset_skip)))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                             num_workers=args.num_workers)

    if not args.save_path:
        save_path = f"{args.output_dir}/{args.experiment_name}/model_ckpt.torch"
    else:
        save_path = args.save_path

    model = eval(args.model)(dataset=args.dataset, device=args.device)
    saved_dict = torch.load(save_path)
    if (args.model == "WideResNetCompat"):
        saved_dict = saved_dict['state_dict']
        for key in list(saved_dict.keys()):
            saved_dict["model.module" + key[1:]] = saved_dict.pop(key)
    if (args.with_denoiser):
        denoiser_model = DnCNN(image_channels=3, depth=17, n_channels=64).to(args.device)
        model.norm = torch.nn.Sequential(denoiser_model, model.norm)
    model.load_state_dict(saved_dict)
    if (args.denoiser_path is not None):
        denoiser_model = DnCNN(image_channels=3, depth=17, n_channels=64).to(args.device)
        denoiser_dict = torch.load(args.denoiser_path)['state_dict']
        denoiser_model.load_state_dict(denoiser_dict)
        model = torch.nn.Sequential(denoiser_model, model)
    model.eval()

    noise = parse_noise_from_args(args, device=args.device, dim=get_dim(args.dataset))

    results = {
        "preds": np.zeros((len(test_dataset), get_num_labels(args.dataset))),
        "labels": np.zeros(len(test_dataset)),
        "prob_lb": np.zeros(len(test_dataset)),
        "preds_nll": np.zeros(len(test_dataset)),
        "radius_l1": np.zeros(len(test_dataset)),
        "radius_l2": np.zeros(len(test_dataset)),
        "radius_linf": np.zeros(len(test_dataset)),
        "time": 0.,
        "time_per" : 0.
    }
    start_time = time.time()
    for i, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):

        x, y = x.to(args.device), y.to(args.device)
        x = rotate_noise.sample(x) if args.rotate else x

        preds = smooth_predict_hard(model, x, noise, args.sample_size_pred,
                                           noise_batch_size=args.noise_batch_size)
        top_cats = preds.probs.argmax(dim=1)
        prob_lb = certify_prob_lb(model, x, top_cats, 0.001, noise,
                                  args.sample_size_cert, noise_batch_size=args.noise_batch_size)

        lower, upper = i * args.batch_size, (i + 1) * args.batch_size
        results["preds"][lower:upper, :] = preds.probs.data.cpu().numpy()
        results["labels"][lower:upper] = y.data.cpu().numpy()
        results["prob_lb"][lower:upper] = prob_lb.cpu().numpy()
        results["radius_l1"][lower:upper] = noise.certify_l1(prob_lb).cpu().numpy()
        results["radius_l2"][lower:upper] = noise.certify_l2(prob_lb).cpu().numpy()
        results["radius_linf"][lower:upper] = noise.certify_linf(prob_lb).cpu().numpy()
        results["preds_nll"][lower:upper] = -preds.log_prob(y).cpu().numpy()
        results["time"] = time.time() - start_time
        results["time_per"] = results["time"]/len(test_dataset)

    save_path = f"{args.output_dir}/{args.experiment_name}"
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    for k, v in results.items():
        np.save(f"{save_path}/{k}.npy", v)

