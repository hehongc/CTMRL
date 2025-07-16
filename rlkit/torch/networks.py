"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F
from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm
import pdb
# import rlkit.torch.transformer as transformer


from einops import rearrange, repeat
# from torch.cuda.amp import autocast
from contextlib import contextmanager

from sklearn.cluster import KMeans

def identity(x):
    return x

class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            batch_attention=False,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            output_activation_half=False,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            use_dropout=False,
            layer_norm_kwargs=None,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        # when output is [mean, var], if output_activation_half is true ,just activate mean, not var
        self.output_activation_half = output_activation_half
        self.layer_norm = layer_norm
        self.use_dropout = use_dropout
        self.fcs = []
        self.layer_norms = []
        self.dropouts = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)
            
            if self.use_dropout:
                dropout_n = nn.Dropout(0.1)
                self.__setattr__("drop_out{}".format(i), dropout_n)
                self.dropouts.append(dropout_n)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)
        
        self.batch_attention = batch_attention
        self.transition_attention = transformer.BatchTransitionAttention(
            hidden=100,
            input_size=input_size,
            output_size=input_size,
            n_layers=3,
            attn_heads=1,
            dropout=0.1
        ) if self.batch_attention else None

    def forward(self, input, return_preactivations=False):
        if self.batch_attention:
            input = self.transition_attention(input)
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
            if self.use_dropout and i < len(self.fcs) - 1:
                h = self.dropouts[i](h)
        preactivation = self.last_fc(h)
        half_output_size = int(self.output_size/2)
        if self.output_activation_half:
            output =  torch.cat([self.output_activation(preactivation[..., :half_output_size]), preactivation[..., half_output_size:]], dim=-1)
        else:
            output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    if there are multiple inputs, concatenate along dim 1
    """

    def forward(self, meta_size=16, batch_size=256, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)

class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class MlpEncoder(Mlp):
    '''
    encode context via MLP
    '''

    def reset(self, num_tasks=1):
        pass


class RecurrentEncoder(FlattenMlp):
    '''
    encode context via recurrent network
    '''

    def __init__(self,
                 *args,
                 **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(ptu.device)))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out[:, -1, :]

        # output layer
        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)


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

    # def forward(self, input):
    #     b, N, dim = input.shape
    #
    #     scores = self.score_func(input.reshape(-1, dim)).reshape(b, N)
    #     scores = F.softmax(scores, dim=-1)
    #
    #     context = scores.unsqueeze(-1).expand_as(input).mul(input)
    #     context_sum = context.sum(1)
    #
    #     return context, context_sum

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

# LISA VQ-VAE
@contextmanager
def null_context():
    yield


def exists(val):
    return val is not None


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


# samples 为输入的数据, 此处输入的应该是用于match的match对象的数据
# num_clusters 为生成的聚类中心的数量
# use_cosine_sim 为衡量距离的方式
# 仅在 init 里面被调用一次 因为init本身就会被且仅被调用一次
def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    # 返回随机选取的tensor作为聚类中心
    means = sample_vectors(samples, num_clusters)

    # 为什么跟循环相关
    for _ in range(num_iters):
        # 余弦相似度衡量距离
        if use_cosine_sim:
            dists = samples @ means.t()
        # 欧氏距离衡量距离
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim=-1)


        # buckets 表示的是匹配的聚类中心的索引 index
        # dist.shape = torch.Size([batch_size, codebook_size])
        # buckets.shape = torch.Size([batch_size, 1])
        buckets = dists.max(dim=-1).indices

        # num_clusters = codebook_size
        # bins 表示为数量统计, 统计samples对于作为聚类中心的means的匹配情况
        bins = torch.bincount(buckets, minlength=num_clusters)
        # bool 判断
        # torch.Size([batch_size,])
        zero_mask = bins == 0
        # 将原本没有匹配/数量统计为0的位置变为1
        # 为了保证 kmeans质心不为0
        # 实际上根本不会有为0的 因为肯定有自我匹配
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        # new_means.shape = torch.Size([batch_size, dim])
        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        # new_means 表示的是全部匹配的分发累积结果
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d=dim), samples)
        # bins_min_clamped 指的是累积的数量
        # new_means 表示的是Tensor的对应质心的索引累积
        # 相当于求均值得到新的聚类中心位置
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        # means 更新
        # 仅更新 zero_mask 中为 True 即有匹配的点
        means = torch.where(zero_mask[..., None], means, new_means)

    # 返回的means和bins都对应最后一次的
    # 分别为聚类中心位置和对应的匹配累积情况
    return means, bins


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


# samples 为输入的数据, 此处输入的应该是用于match的match对象的数据
# num 为生成的聚类中心的数量
def sample_vectors(samples, num):
    # 事先获取输入的tensor的数量
    num_samples, device = samples.shape[0], samples.device

    # 如果输入的tensor的数量比指定需要生成的聚类中心的数量要多则随机采样
    if num_samples >= num:
        # 不重复采样
        indices = torch.randperm(num_samples, device=device)[:num]
    # 如果输入的tensor的数量比指定需要生成的聚类中心的数量要少则重复采样
    else:
        # 允许重复采样
        indices = torch.randint(0, num_samples, (num,), device=device)

    # 返回的是聚类中心 list/tensor
    # 且聚类中心是随机选取的
    # 返回的 indices 为 sample index
    return samples[indices]

# # self.cluster_size 表示 Codebook 里面 Code 的数量
# embed_onehot 为当前 Match 的情况
# self.decay = 0.8
# ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
def ema_inplace(moving_avg, new, decay):
    # moving_avg 取 self.cluster_size 则表示的是一个初始全部为0的tensor
    # moving_avg 实际上取的是在聚类中心上匹配累积数量的情况
    # new 指的是当前在聚类中心上匹配累积数量的情况
    # 先前累积的数量 * decay + 当前累积数量的情况 * (1 - decay)
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


# Build info_nce function as loss function.
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
        # if self.training:
        #     quantize = x + (quantize - x).detach()

        quantize = self.project_out(quantize)

        if need_transpose:
            quantize = quantize.transpose(1, 2)

        return quantize

    def forward(self, x):
        x_shape = x.shape
        x_device = x.device

        # 完全不考虑输入的数据为 image 的情况
        # channel_last 表示的是维度正常的情况
        # 如果 not self.channel_last 表示维度的分布不正常 需要调换维度
        need_transpose = not self.channel_last and not self.accept_image_fmap

        # 此处设置为默认调换最后两个维度
        if need_transpose:
            x = x.transpose(1, 2)

        x = self.project_in(x)

        # context = null_context() if not self.quantize_full_precision else autocast(enabled=False)

        context = null_context()

        with context:
            # 将处理后的 x 输入到 self._codebook 中进行匹配处理
            # quantize 表示的是匹配的embedding value
            # embed_ind 表示的是匹配的embedding index
            # dist 表示匹配情况 为近似值矩阵
            quantize, embed_ind, dist = self._codebook(x)

        # if self.training:
        #     quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.], device=x_device, requires_grad=self.training)

        if self.training:
            # True
            if self.commitment_weight > 0:
                # cpc = False
                if self.cpc:
                    # quantize 表示的是与输入的 x 进行匹配的 code
                    # x 表示 encoder 实际输出的 embedding
                    # embed_ind 表示的quantize 在整个 codebook 中的排序情况
                    # dist 表示的是匹配情况 即近似值矩阵
                    commit_loss = info_nce(quantize.detach(), x, embed_ind, dist)
                else:
                    # commit_loss = F.mse_loss(quantize.detach(), x)
                    commit_loss = F.mse_loss(quantize, x)

                loss = loss + commit_loss * self.commitment_weight

            # False
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

        # init_fn = torch.randn if not kmeans_init else torch.zeros
        init_fn = torch.zeros
        embed = init_fn(self.codebook_size, self.input_dim)

        self.register_buffer('initted', torch.Tensor([not self.kmeans_init]))
        # 初始化 cluster_size 为一个一维 tensor
        self.register_buffer('cluster_size', torch.zeros(self.codebook_size))
        # self.register_buffer('embed', embed)
        self.register_buffer('embed_avg', embed.clone())

        self.embed = nn.ParameterList([
            nn.Parameter(embed, requires_grad=True)
        ])

    @torch.jit.ignore
    def init_embed_(self, data):
        # init_embed_ 如果被调用的话会触发一次
        # print("initted: ", self.initted)
        if self.initted:
            return
        # exit()
        # 输入的data指的就是输入的match对象的数据
        # self.codebook_size 表示生成的聚类中心的数量
        # self.kmeans_iters 为超参数 = 10
        # embed 表示的是聚类中心的位置
        # cluster_size 表示的是聚类中心上的累积数量
        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        # 初始化情况下 self.embed_avg = self.embed
        self.embed_avg.data.copy_(embed.clone())
        # cluster_size 表示的是聚类中心上的累积数量
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, samples, mask):
        # mask 为 bool 矩阵 表示的是是否有满足match阈值的聚类中心
        modified_codebook = torch.where(
            mask[..., None],
            # sample_vectors 表示为随机采样
            sample_vectors(samples, self.codebook_size),
            self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):

        # False
        if self.threshold_ema_dead_code == 0:
            return

        # 有其中一个 Code 被 self.threshold_ema_dead_code 数量以上的 Match 了
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        # reshape
        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, mask=expired_codes)

    # def init_VQ(self, collected_context_embeddings):
    #     collected_context_embeddings = collected_context_embeddings.detach().cpu().numpy()
    #     kmeans = KMeans(n_clusters=self.codebook_size, random_state=0).fit(collected_context_embeddings)
    #     cluster_centers = kmeans.cluster_centers_
    #     cluster_centers_tensor = torch.tensor(cluster_centers).cuda()
    #
    #     self.embed = cluster_centers_tensor
    #     self.embed_avg.data.copy_(cluster_centers_tensor.clone())

    def init_VQ(self, collected_context_embeddings):
        collected_context_embeddings = collected_context_embeddings.reshape(-1, self.input_dim)
        embed, cluster_size = kmeans(collected_context_embeddings, self.codebook_size, self.kmeans_iters, use_cosine_sim=True)
        with torch.no_grad():
            self.embed[0].copy_(embed)
            # self.embed.data.copy_(embed)
            self.embed_avg.data.copy_(embed.clone())
            self.cluster_size.data.copy_(cluster_size)

    def finetune_VQ(self, collected_context_embeddings):
        collected_context_embeddings = collected_context_embeddings.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.codebook_size, random_state=0).fit(collected_context_embeddings)
        cluster_centers = kmeans.cluster_centers_
        cluster_centers_tensor = torch.tensor(cluster_centers).cuda()


    def Kmeans_update_VQ(self, collected_context_embeddings):

        x, _ = kmeans(collected_context_embeddings, self.codebook_size, self.kmeans_iters,
                                     use_cosine_sim=True)

        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, '... d -> (...) d')
        # 欧氏距离
        # embed = self.embed[0].t()
        # dist = -(
        #         flatten.pow(2).sum(1, keepdim=True)
        #         - 2 * flatten @ embed
        #         + embed.pow(2).sum(0, keepdim=True)
        # )

        # 余弦相似度 + nn.ParameterList
        embed = self.embed[0]
        flatten_norm = flatten / torch.norm(flatten, dim=-1, keepdim=True)
        embed_norm = embed / torch.norm(embed, dim=-1, keepdim=True)
        dist = torch.mm(flatten_norm, embed_norm.transpose(0, 1))

        embed_ind = dist.max(dim=-1).indices
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        # quantize = F.embedding(embed_ind, self.embed[0])

        # training
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
        # 欧氏距离
        # embed = self.embed[0].t()
        # 余弦相似度 + nn.ParameterList
        embed = self.embed[0]

        # self.init_embed_(flatten)

        # 使用欧氏距离衡量距离
        # [task_num, codebook_size]
        # dist = -(
        #         flatten.pow(2).sum(1, keepdim=True)
        #         - 2 * flatten @ embed
        #         + embed.pow(2).sum(0, keepdim=True)
        # )

        # 使用余弦相似度衡量距离
        flatten_norm = flatten / torch.norm(flatten, dim=-1, keepdim=True)
        embed_norm = embed / torch.norm(embed, dim=-1, keepdim=True)
        dist = torch.mm(flatten_norm, embed_norm.transpose(0, 1))

        # [task_num, codebook_size]
        dist = F.softmax(dist, dim=-1)

        # [task_num, codebook_dim]
        match_code = dist @ self.embed[0]

        return match_code

    def forward(self, x):
        # 获取 x 相关的信息
        # shape of input x is [task_num, codebook_dim]
        # shape = [task_num, codebook_dim]
        shape, dtype = x.shape, x.dtype
        # 将 x 展平处理 输出为一个一维的 tensor
        # flatten.shape = [task_num, codebook_dim]
        flatten = rearrange(x, '... d -> (...) d')
        # self.embed 表示的就是 codebook
        # (codebook_size, codebook_dim) -> (codebook_dim, codebook_size)
        # 欧氏距离
        # embed = self.embed[0].t()
        # 余弦相似度 + nn.ParameterList
        embed = self.embed[0]

        # 初始化构建 Codebook
        # 会且最多会被执行一次
        # self.init_embed_(flatten)

        # 计算 distance
        # 计算的是欧氏距离
        # 用距离取负数处理 得到的是相似度
        # flatten.pow(2).sum(1, keepdim=True).shape = [task_num, 1]
        # embed.pow(2).sum(0, keepdim=True).shape = [1, codebook_size]
        # flatten @ embed.shape = [task_num, codebook_size]
        # dist = -(
        #         flatten.pow(2).sum(1, keepdim=True)
        #         - 2 * flatten @ embed
        #         + embed.pow(2).sum(0, keepdim=True)
        # )

        # 使用余弦相似度衡量距离
        flatten_norm = flatten / torch.norm(flatten, dim=-1, keepdim=True)
        embed_norm = embed / torch.norm(embed, dim=-1, keepdim=True)
        dist = torch.mm(flatten_norm, embed_norm.transpose(0, 1))

        # 选择 max 下的 indices
        # 选取的是最接近的一个点对应的索引值 idx
        # dist.shape:  torch.Size([16, 16])
        # dist.shape: torch.Size([batch_size, codebook_size])
        # print("dist.shape: ", dist.shape)

        # embed_ind = torch.Size([batch_size, 1])
        embed_ind = dist.max(dim=-1).indices

        # torch.Size([batch_size, codebook_size])
        # embed_onehot 表示的是索引 one-hot
        # 每一行表示的是当前的 tensor 索引到哪一个 code
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        # F.embedding 起到 word embedding 的作用
        # quantize 表示的是实际进行 match 的
        quantize = F.embedding(embed_ind, self.embed[0])

        # True
        # 如何更新 codebook
        if self.training:
            # def ema_inplace(moving_avg, new, decay):
            #     moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))
            # ema_inplace函数表示更新过程
            # 更新过程将被表示为衰减因子 decay 影响下的累加

            # embed_onehot 为当前 Match 的情况
            # embed_onehot.sum(0) = torch.Size([codebook_size, ])
            # self.decay = 0.8
            # 这部分累积是累积在 self.cluster_size 上, 累积的是match的数量
            # self.register_buffer('cluster_size', torch.zeros(self.codebook_size))
            # self.cluster_size 表示的本身就是在聚类中心上累积数量的情况
            # ema_inplace 这部分单纯对数量进行累积
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)

            # flatten.shape = [batch_size, codebook_dim]
            # flatten.t.shape = [codebook_dim, batch_size]
            # embed_onehot.shape = torch.Size([batch_size, codebook_size])
            # embed_sum.shape = torch.Size([codebook_dim, codebook_size])
            # 构建到聚类中心上的 Tensor的累积
            embed_sum = flatten.t() @ embed_onehot
            # 同样累积
            # embed_sum.t().shape = torch.Size([codebook_size, codebook_dim])
            # self.embed_avg.shape = torch.Size([codebook_size, codebook_dim])
            # embed_avg 表示的是全局的累积
            # ema_inplace这部分进行的是 Tensor的累积
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)

            # self.cluster_size 表示的是累积的 Match 次数
            # 但是不清楚是当前累积的还是长期累积的
            # self.codebook_size
            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()
            # self.embed_avg.shape = torch.Size([codebook_size, codebook_dim])
            #
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            # 更新 embed
            # self.embed.data.copy_(embed_normalized)
            with torch.no_grad():
                self.embed[0].copy_(embed_normalized)
            # 也会涉及到更新 embed
            # self.expire_codes_(x)

        # quantize 表示的是匹配的embedding value
        # embed_ind 表示的是匹配的embedding index
        # dist 表示匹配情况 为近似值矩阵
        return quantize, embed_ind, dist


def entropy(codes, select_codes, vq_embeddings):
    with torch.no_grad():
        N, D = codes.shape
        vq_embeddings = vq_embeddings.reshape(-1, 1, D)

        # 欧氏距离
        # embed = codes.t()
        # 余弦相似度
        embed = codes
        flatten = rearrange(vq_embeddings, '... d -> (...) d')

        # 欧氏距离
        # distance = -(
        #         flatten.pow(2).sum(1, keepdim=True)
        #         - 2 * flatten @ embed
        #         + embed.pow(2).sum(0, keepdim=True)
        # )

        # 余弦相似度
        flatten_norm = flatten / torch.norm(flatten, dim=-1, keepdim=True)
        embed_norm = embed / torch.norm(embed, dim=-1, keepdim=True)
        distance = torch.mm(flatten_norm, embed_norm.transpose(0, 1))

        cond_probs = torch.softmax(distance / 2, dim=1)

        probs = cond_probs.mean(dim=0)
        entropy = (-torch.log2(probs) * probs).sum()
        cond_entropy = (-torch.log2(cond_probs) * cond_probs).sum(1).mean(0)

        return (entropy, cond_entropy)




