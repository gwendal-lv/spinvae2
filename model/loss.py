"""
Contains specialized losses, e.g. contrastive or attribute-based latent losses.
"""

import torch
import torch.nn.functional as F


def s2_vae_latent_loss(z_mu: torch.Tensor, target_attributes: torch.Tensor, normalize_latent: bool):
    """
    https://arxiv.org/abs/1905.01258 ICLR20, https://arxiv.org/abs/2108.01450 (ISMIR21)

    :param normalize_latent: If False, Sum over latent dimensions (more attributes -> more regularization)
        instead of averaging. An average is always computed along the minibatch dimension.
    """
    logsigm_mu = F.logsigmoid(z_mu)
    normalized_a = torch.sigmoid(target_attributes)
    BCE_loss = -((normalized_a * logsigm_mu) + (1.0 - normalized_a) * (-z_mu + logsigm_mu))
    BCE_loss = torch.sum(BCE_loss, dim=1) if not normalize_latent else torch.mean(BCE_loss, dim=1)
    return BCE_loss.mean()


def ar_vae_latent_loss(z_mu: torch.Tensor, target_attributes: torch.Tensor, normalize: bool, delta: float):
    """
    https://arxiv.org/abs/2004.05485 , https://arxiv.org/abs/2108.01450
    """
    M, N_A = z_mu.shape[0], target_attributes.shape[1]  # minibatch size, number of attributes
    # attributes distance matrix
    assert tuple(target_attributes.T.shape) == (N_A, M)
    expanded_rows_attributes = target_attributes.T.unsqueeze(dim=2).expand(N_A, M, M)
    expanded_cols_attributes = expanded_rows_attributes.transpose(1, 2)
    attributes_distance_matrices = expanded_rows_attributes - expanded_cols_attributes
    # compute null attr diff mask: we won't use the gradient when attributes are very close
    # (the difference's sign becomes meaningless)  FIXME maybe useless using torch.sign ????
    gradient_mask = torch.isclose(attributes_distance_matrices, torch.zeros_like(attributes_distance_matrices))
    gradient_mask = 1.0 - gradient_mask.float()
    # latent values (representation r) distance matrix
    assert tuple(z_mu.T.shape) == (N_A, M)
    expanded_rows_latent = z_mu.T.unsqueeze(dim=2).expand(N_A, M, M)
    expanded_cols_latent = expanded_rows_latent.transpose(1, 2)
    latent_distance_matrices = expanded_rows_latent - expanded_cols_latent
    # TODO null loss if attributes were equal
    sign_based_error = torch.abs(torch.tanh(delta * latent_distance_matrices)
                                 - torch.sign(attributes_distance_matrices))
    # Latent dimension (now in first dim, after tranpose) can be summed or averaged:
    sign_based_error = torch.sum(sign_based_error, dim=0) if not normalize else torch.mean(sign_based_error, dim=0)
    # Normalizing: we always average over the last dimension (which is an artificial appended minibatch dim),
    #   and over the actual minibatch dim which is dim 1 (after transposition).
    return sign_based_error.mean()


