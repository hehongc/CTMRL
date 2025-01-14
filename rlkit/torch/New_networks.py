import torch
from torch import nn as nn
from torch.nn import functional as F
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule

from torch.nn.utils import weight_norm
import torch.nn.init as init
import pywt
import numpy as np
import numpy

from rlkit.torch.networks import Mlp

from einops import rearrange, repeat
from contextlib import contextmanager

from sklearn.cluster import KMeans




class SelfAttnEncoder(PyTorchModule):
    def __init__(self,
                 input_dim,
                 num_output_mlp=0,
                 task_gt_dim=5):
        super(SelfAttnEncoder, self).__init__()

        self.input_dim = input_dim
        self.score_func = nn.Linear(input_dim, 1)

        self.num_output_mlp = num_output_mlp

        if num_output_mlp > 0:
            self.output_mlp = Mlp(
                input_size=input_dim,
                output_size=task_gt_dim,
                hidden_sizes=[200 for i in range(num_output_mlp - 1)]
            )



    def forward(self, input):
        if len(input.shape) == 3:
            b, N, dim = input.shape
            scores = self.score_func(input.reshape(-1, dim)).reshape(b, N, -1)
        elif len(input.shape) == 2:
            b, dim = input.shape
            scores = self.score_func(input.reshape(-1, dim)).reshape(b, -1)

        return scores

    def compute_softmax_result(self, input_1, score_1, input_2, score_2):
        scores = torch.cat([score_1, score_2], dim=-1).cuda()
        softmax_scores = F.softmax(scores, dim=-1).cuda()

        if len(input_1.shape) == 3:
            t = input_1.shape[0]
            b = input_1.shape[1]
            softmax_score_1 = softmax_scores[:, :, 0].reshape(t, b, 1)
            combine_input_1 = input_1 * softmax_score_1
            softmax_score_2 = softmax_scores[:, :, 1].reshape(t, b, 1)
            combine_input_2 = input_2 * softmax_score_2
            combine_output = combine_input_1 + combine_input_2
        elif len(input_1.shape) == 2:
            t = input_1.shape[0]
            softmax_score_1 = softmax_scores[:, 0].reshape(t, 1)
            combine_input_1 = input_1 * softmax_score_1
            softmax_score_2 = softmax_scores[:, 1].reshape(t, 1)
            combine_input_2 = input_2 * softmax_score_2
            combine_output = combine_input_1 + combine_input_2

        return combine_output, softmax_score_1, softmax_score_2


class NewSelfAttnEncoder(PyTorchModule):
    def __init__(self,
                 input_dim,
                 head_num,
                 ):
        super(NewSelfAttnEncoder, self).__init__()

        self.input_dim = input_dim
        self.head_num = head_num

        self.score_func = nn.Linear(input_dim, 1)


    def forward(self, input):
        b, N = input.shape[0], input.shape[1]
        dim = input.shape[-1]

        split_part = []
        for _ in range(self.head_num):
            split_part.append(1)
        each_context = torch.split(input, split_part, dim=-2)

        context_score = None
        for each_context_part in each_context:
            # (b, N, 1)
            each_context_part_score = self.score_func(each_context_part.squeeze(-2))
            if context_score == None:
                context_score = each_context_part_score
            else:
                # (b, N, head_num)
                context_score = torch.cat([context_score, each_context_part_score], dim=-1)

        # (b, N, head_num)
        context_score = F.softmax(context_score, dim=-1)
        # (b, N, 1) * head_num
        context_score = torch.split(context_score, split_part, dim=-1)

        context_output = None
        for i in range(self.head_num):
            current_context_score = context_score[i].expand_as(each_context[i].squeeze(-2))
            current_context = current_context_score * each_context[i].squeeze(-2)
            if context_output == None:
                context_output = current_context
            else:
                context_output += current_context

        # (b, N, dim)
        return context_output


class CVAE(nn.Module):
    def __init__(self,
                 hidden_size=64,
                 num_hidden_layers=1,
                 z_dim=20,
                 action_dim=5,
                 state_dim=2,
                 reward_dim=1,
                 use_ib=False,
                 ):

        super(CVAE, self).__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.reward_dim = reward_dim
        self.z_dim = z_dim

        self.use_ib = use_ib

        if self.use_ib:
            self.encoder = Mlp(
                input_size=self.state_dim * 2 + self.action_dim + self.reward_dim,
                output_size=self.z_dim * 2,
                hidden_sizes=[hidden_size for i in range(num_hidden_layers)]
            )
        else:
            self.encoder = Mlp(
                input_size=self.state_dim * 2 + self.action_dim + self.reward_dim,
                output_size=self.z_dim,
                hidden_sizes=[hidden_size for i in range(num_hidden_layers)]
            )

        self.decoder = Mlp(
            input_size=self.z_dim + self.state_dim + self.action_dim,
            output_size=self.state_dim + self.reward_dim,
            hidden_sizes=[hidden_size for i in range(num_hidden_layers)]
        )






@contextmanager
def null_context():
    yield


def exists(val):
    return val is not None


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices

        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def info_nce(code, embeds, pos_indices, dist):
    B, T, D = code.shape
    dist = dist.reshape(B, T, -1)

    p = torch.log_softmax(dist / 2, dim=-1)
    loss = -p
    loss = loss[pos_indices]

    return loss.mean()


def orthgonal_loss_fn(t):
    n = t.shape[0]
    normed_codes = l2norm(t)
    identity = torch.eye(n, device=t.device)
    cosine_sim = torch.einsum('i d, j d -> i j', normed_codes, normed_codes)

    return ((cosine_sim - identity) ** 2).sum() / (n ** 2)


class VectorQuantize(PyTorchModule):
    def __init__(self, input_dim,
                 codebook_size,
                 code_dim,
                 decay=0.8,
                 eps=1e-5,
                 kmeans_init=False,
                 kmeans_iters=10,
                 use_cosine_sim=False,
                 threshold_ema_dead_code=0,
                 channel_last=True,
                 accept_image_fmap=False,
                 commitment_weight=None,
                 commitment=1.,  # deprecate in next version, turn off by default
                 orthogonal_reg_weight=0.,
                 orthogonal_reg_active_codes_only=False,
                 orthogonal_reg_max_codes=None,
                 quantize_full_precision=False,
                 cpc=False,
                 ):
        super(VectorQuantize, self).__init__()

        self.codebook_size = codebook_size
        self.input_dim = input_dim
        self.code_dim = code_dim

        self.requires_projection = self.code_dim != self.input_dim

        self.project_in = nn.Linear(self.input_dim, code_dim) if self.requires_projection else nn.Identity()
        self.project_out = nn.Linear(self.code_dim, input_dim) if self.requires_projection else nn.Identity()

        self.eps = eps
        self.commitment_weight = commitment_weight
        self.decay = decay

        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

        self.cpc = cpc
        self.quantize_full_precision = quantize_full_precision
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.codebook_class = EuclideanCodebook

        self._codebook = self.codebook_class(
            input_dim=self.code_dim,
            codebook_size=self.codebook_size,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            decay=self.decay,
            eps=self.eps,
            threshold_ema_dead_code=self.threshold_ema_dead_code
        )

    @property
    def codebook(self):
        return self._codebook.embed[0]

    def init_VQ(self, collected_context_embeddings):
        self._codebook.init_VQ(collected_context_embeddings)

    def Kmeans_update_VQ(self, collected_context_embeddings):
        self._codebook.Kmeans_update_VQ(collected_context_embeddings)

    def forward_for_test(self, x):
        x_shape = x.shape
        x_device = x.device
        need_transpose = not self.channel_last and not self.accept_image_fmap

        if need_transpose:
            x = x.transpose(1, 2)

        x = self.project_in(x)

        context = null_context()
        with context:
            quantize = self._codebook.forward_for_test(x)

        quantize = self.project_out(quantize)

        if need_transpose:
            quantize = quantize.transpose(1, 2)

        return quantize

    def forward(self, x):
        x_shape = x.shape
        x_device = x.device

        need_transpose = not self.channel_last and not self.accept_image_fmap

        if need_transpose:
            x = x.transpose(1, 2)

        x = self.project_in(x)
        context = null_context()

        with context:
            quantize, embed_ind, dist = self._codebook(x)

        loss = torch.tensor([0.], device=x_device, requires_grad=self.training)

        if self.training:
            if self.commitment_weight > 0:
                if self.cpc:
                    commit_loss = info_nce(quantize.detach(), x, embed_ind, dist)
                else:
                    commit_loss = F.mse_loss(quantize, x)

                loss = loss + commit_loss * self.commitment_weight

            if self.orthogonal_reg_weight > 0:
                codebook = self.codebook

                if self.orthogonal_reg_active_codes_only:
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[unique_code_ids]

                num_codes = codebook.shape[0]
                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device=x_device)[: self.orthogonal_reg_max_codes]
                    codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthgonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        quantize = self.project_out(quantize)

        if need_transpose:
            quantize = quantize.transpose(1, 2)

        return quantize, embed_ind, loss


class EuclideanCodebook(PyTorchModule):
    def __init__(self,
                 input_dim,
                 codebook_size,
                 kmeans_init=False,
                 kmeans_iters=10,
                 decay=0.8,
                 eps=1e-5,
                 threshold_ema_dead_code=2
                 ):
        super(EuclideanCodebook, self).__init__()

        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.decay = decay
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code

        init_fn = torch.zeros
        embed = init_fn(self.codebook_size, self.input_dim)

        self.register_buffer('initted', torch.Tensor([not self.kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(self.codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.embed = nn.ParameterList([
            nn.Parameter(embed, requires_grad=True)
        ])

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None],
            sample_vectors(samples, self.codebook_size),
            self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):

        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask=expired_codes)

    def init_VQ(self, collected_context_embeddings):
        collected_context_embeddings = collected_context_embeddings.reshape(-1, self.input_dim)
        embed, cluster_size = kmeans(collected_context_embeddings, self.codebook_size, self.kmeans_iters, use_cosine_sim=True)
        with torch.no_grad():
            self.embed[0].copy_(embed)
            self.embed_avg.data.copy_(embed.clone())
            self.cluster_size.data.copy_(cluster_size)

    def finetune_VQ(self, collected_context_embeddings):
        collected_context_embeddings = collected_context_embeddings.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.codebook_size, random_state=0).fit(collected_context_embeddings)
        cluster_centers = kmeans.cluster_centers_
        cluster_centers_tensor = torch.tensor(cluster_centers).cuda()


    def Kmeans_update_VQ(self, collected_context_embeddings):

        x, _ = kmeans(collected_context_embeddings, self.codebook_size, self.kmeans_iters,
                                     use_cosine_sim=False)

        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')

        embed = self.embed[0]
        flatten_norm = flatten / torch.norm(flatten, dim=-1, keepdim=True)
        embed_norm = embed / torch.norm(embed, dim=-1, keepdim=True)
        dist = torch.mm(flatten_norm, embed_norm.transpose(0, 1))

        embed_ind = dist.max(dim=-1).indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])

        ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
        embed_sum = flatten.t() @ embed_onehot
        ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
        cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        with torch.no_grad():
            self.embed[0].copy_(embed_normalized)



    def forward_for_test(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        embed = self.embed[0]

        flatten_norm = flatten / torch.norm(flatten, dim=-1, keepdim=True)
        embed_norm = embed / torch.norm(embed, dim=-1, keepdim=True)
        dist = torch.mm(flatten_norm, embed_norm.transpose(0, 1))

        dist = F.softmax(dist, dim=-1)

        match_code = dist @ self.embed[0]

        return match_code

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        embed = self.embed[0]

        flatten_norm = flatten / torch.norm(flatten, dim=-1, keepdim=True)
        embed_norm = embed / torch.norm(embed, dim=-1, keepdim=True)
        dist = torch.mm(flatten_norm, embed_norm.transpose(0, 1))

        embed_ind = dist.max(dim=-1).indices

        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = F.embedding(embed_ind, self.embed[0])

        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            with torch.no_grad():
                self.embed[0].copy_(embed_normalized)
        return quantize, embed_ind, dist


def entropy(codes, select_codes, vq_embeddings):
    with torch.no_grad():
        N, D = codes.shape
        vq_embeddings = vq_embeddings.reshape(-1, 1, D)

        embed = codes
        flatten = rearrange(vq_embeddings, '... d -> (...) d')
        flatten_norm = flatten / torch.norm(flatten, dim=-1, keepdim=True)
        embed_norm = embed / torch.norm(embed, dim=-1, keepdim=True)
        distance = torch.mm(flatten_norm, embed_norm.transpose(0, 1))

        cond_probs = torch.softmax(distance / 2, dim=1)

        probs = cond_probs.mean(dim=0)
        entropy = (-torch.log2(probs) * probs).sum()
        cond_entropy = (-torch.log2(cond_probs) * cond_probs).sum(1).mean(0)

        return (entropy, cond_entropy)

