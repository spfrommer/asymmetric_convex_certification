import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from statsmodels.stats.proportion import proportion_confint


def direct_train_log_lik(model, x, y, noise, sample_size=16):
    """
    Log-likelihood for direct training (numerically stable with logusmexp trick).
    """
    samples_shape = torch.Size([x.shape[0], sample_size]) + x.shape[1:]
    samples = x.unsqueeze(1).expand(samples_shape)
    samples = samples.reshape(torch.Size([-1]) + samples.shape[2:])
    samples = noise.sample(samples)
    thetas = model(samples).view(x.shape[0], sample_size, -1)
    return torch.logsumexp(thetas[torch.arange(x.shape[0]), :, y] - \
                           torch.logsumexp(thetas, dim=2), dim=1) - \
           torch.log(torch.tensor(sample_size, dtype=torch.float, device=x.device))

def smooth_predict_soft(model, x, noise, sample_size=64, noise_batch_size=512, softmax=True):
    """
    Make soft predictions for a model smoothed by noise.

    Returns
    -------
    predictions: Categorical, probabilities for each class returned by soft smoothed classifier
    """
    counts = None
    num_samples_left = sample_size

    while num_samples_left > 0:

        shape = torch.Size([x.shape[0], min(num_samples_left, noise_batch_size)]) + x.shape[1:]
        samples = x.unsqueeze(1).expand(shape)
        samples = samples.reshape(torch.Size([-1]) + samples.shape[2:])
        samples = noise.sample(samples.view(len(samples), -1)).view(samples.shape)
        logits = model(samples).view(shape[:2] + torch.Size([-1]))
        if counts is None:
            counts = torch.zeros(x.shape[0], logits.shape[-1], dtype=torch.float, device=x.device)

        if softmax:
            counts += F.softmax(logits, dim=-1).mean(dim=1)
        else:
            counts += logits.sum(dim=1) / sample_size

        num_samples_left -= noise_batch_size

    if softmax:
        return Categorical(probs=counts)
    else:
        return counts

def smooth_predict_hard(model, x, noise, sample_size=64, noise_batch_size=512, raw_count=False):
    """
    Make hard predictions for a model smoothed by noise.

    Returns
    -------
    predictions: Categorical, probabilities for each class returned by hard smoothed classifier
    """
    counts = None
    num_samples_left = sample_size

    while num_samples_left > 0:

        shape = torch.Size([x.shape[0], min(num_samples_left, noise_batch_size)]) + x.shape[1:]
        samples = x.unsqueeze(1).expand(shape)
        samples = samples.reshape(torch.Size([-1]) + samples.shape[2:])
        samples = noise.sample(samples.view(len(samples), -1)).view(samples.shape)
        logits = model(samples).view(shape[:2] + torch.Size([-1]))
        top_cats = torch.argmax(logits, dim=2)
        if counts is None:
            counts = torch.zeros(x.shape[0], logits.shape[-1], dtype=torch.float, device=x.device)
        counts += F.one_hot(top_cats, logits.shape[-1]).float().sum(dim=1)
        num_samples_left -= noise_batch_size

    if raw_count:
        return counts.squeeze()

    return Categorical(probs=counts)

def smooth_predict_hard_derandomized(model, x, noise,  noise_batch_size=512):
    """
    Make hard predictions for a model smoothed by noise.

    Returns
    -------
    predictions: Categorical, probabilities for each class returned by hard smoothed classifier
    """
    counts = None

    num_samples_left = noise.total_samples_count()

    while num_samples_left > 0:

        start_sample = noise.total_samples_count() - num_samples_left
        num_samples_this_round =  min(num_samples_left, noise_batch_size)
        end_sample = start_sample + num_samples_this_round
        shape = torch.Size([x.shape[0], num_samples_this_round ]) + x.shape[1:]
        flattened_shape = torch.Size([x.shape[0] * num_samples_this_round,]) + x.shape[1:]
        samples = noise.sample_all_deterministic(x.view(x.shape[0], -1),start_sample = start_sample, end_sample = end_sample).view(flattened_shape)
        logits = model(samples).view(shape[:2] + torch.Size([-1]))
        top_cats = torch.argmax(logits, dim=2)
        if counts is None:
            counts = torch.zeros(x.shape[0], logits.shape[-1], dtype=torch.float, device=x.device)
        counts += F.one_hot(top_cats, logits.shape[-1]).float().sum(dim=1)
        num_samples_left -= noise_batch_size

    return counts

def smooth_predict_hard_derandomized_subset(model, x, noise, total_samples, noise_batch_size=512):
    counts = None

    num_samples_left = total_samples

    while num_samples_left > 0:
        num_samples_this_round =  min(num_samples_left, noise_batch_size)
        shape = torch.Size([x.shape[0], num_samples_this_round ]) + x.shape[1:]
        flattened_shape = torch.Size([x.shape[0] * num_samples_this_round,]) + x.shape[1:]
        samples = noise.sample_random_subset(x.view(x.shape[0], -1), num_samples_this_round).view(flattened_shape)
        logits = model(samples).view(shape[:2] + torch.Size([-1]))
        top_cats = torch.argmax(logits, dim=2)
        if counts is None:
            counts = torch.zeros(x.shape[0], logits.shape[-1], dtype=torch.float, device=x.device)
        counts += F.one_hot(top_cats, logits.shape[-1]).float().sum(dim=1)
        num_samples_left -= noise_batch_size

    return counts

def certify_prob_lb(model, x, top_cats, alpha, noise, sample_size=10**5, noise_batch_size=512, sample_size_scale=1):
    """
    Certify a probability lower bound (rho).

    Returns
    -------
    prob_lb: n-length tensor of floats
    """
    preds = smooth_predict_hard(model, x, noise, sample_size, noise_batch_size)
    top_probs = preds.probs.gather(dim=1, index=top_cats.unsqueeze(1)).detach().cpu()
    lower, _ = proportion_confint(top_probs * sample_size * sample_size_scale, sample_size * sample_size_scale, alpha=alpha, method="beta")
    lower = torch.tensor(lower.squeeze(), dtype=torch.float)
    return lower

#def certify_smoothed(model, x, top_cats, alpha, noise, adv, sample_size=10**5, noise_batch_size=512):
#    """
#    Certify a smoothed model, given the top categories to certify for.
#
#    Returns
#    -------
#    lower: n-length tensor of floats, the probability lower bounds
#    radius: n-length tensor
#    """
#    preds = smooth_predict_hard(model, x, noise, sample_size, noise_batch_size)
#    top_probs = preds.probs.gather(dim=1, index=top_cats.unsqueeze(1)).detach().cpu()
#    lower, _ = proportion_confint(top_probs * sample_size, sample_size, alpha=alpha, method="beta")
#    lower = torch.tensor(lower.squeeze(), dtype=torch.float)
#    return lower, noise.certify(lower, adv=adv)
#
