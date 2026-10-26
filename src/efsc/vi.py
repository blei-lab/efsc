"""Mean-field black-box variational inference utilities for EFSC.

This module constructs the Monte Carlo negative-ELBO objective for a diagonal
Gaussian variational distribution and optimizes its mean and log-variance
parameters using Adam.

Author:
    Hector Rodriguez-Deniz

Affiliation:
    Data Science Institute, Columbia University

Date:
    August 2026

License:
    MIT
"""

import time
import torch


def make_negative_elbo(log_joint, dim, device, n_samples=2):
    """
    Create a generic mean-field Gaussian BBVI objective.

    Parameters
    ----------
    log_joint : callable
        Function log_joint(theta) -> scalar torch.Tensor.
    dim : int
        Dimension of latent parameter theta.
    device : torch.device
        Torch device.
    n_samples : int
        Number of Monte Carlo samples for the ELBO estimate.
    """

    def negative_elbo(params):
        """Estimate the negative ELBO from reparameterized Gaussian draws."""

        d = len(params)
        d2 = d // 2

        mean = params[:d2]
        log_s2 = params[d2:]

        sd = torch.sqrt(torch.exp(log_s2))

        z = torch.randn((n_samples, dim), device=device)
        theta_samples = mean[None, :] + sd[None, :] * z

        q = torch.distributions.MultivariateNormal(
            mean,
            torch.diag(torch.exp(log_s2)),
        )

        elbo = 0.0

        for s in range(n_samples):
            theta_s = theta_samples[s, :]
            elbo = elbo + log_joint(theta_s) - q.log_prob(theta_s)

        return -elbo / n_samples

    return negative_elbo


def fit_vi(
    negative_elbo,
    init_params,
    lr=0.1,
    num_epochs=3000,
    print_every=1000,
):
    """
    Optimize a negative ELBO using Adam.
    """

    params = init_params.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([params], lr=lr)
    loss_history = []

    start_time = time.perf_counter()

    for epoch in range(num_epochs):
        loss = negative_elbo(params)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if print_every is not None and (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    elapsed_time = time.perf_counter() - start_time

    return params.detach(), loss_history, elapsed_time
