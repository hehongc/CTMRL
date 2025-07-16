import numpy as np

import torch
import copy
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
import pdb
from rlkit.torch.networks import entropy


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class PEARLAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 # context_discriminator,
                 VectorQuantizeCodebook,
                 policy,
                 **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder
        self.policy = policy

        self.VectorQuantizeCodebook = VectorQuantizeCodebook
        self.use_VQ = False

        self.recurrent = kwargs['recurrent']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.collected_context_embeddings = None
        self.select_codes_score = None
        self.z_means_score = None
        # self.params = None
        self.indices = None
        self.code = None

        self.W = nn.ParameterList(
            [nn.Parameter(torch.rand(self.latent_dim, self.latent_dim), requires_grad=True)]).cuda()

        self.clear_z()

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var

        self.commitment_loss = None
        self.entropies = None
        self.context = None
        self.code = None
        # self.select_codes_score = None
        # self.z_means_score = None
        # self.params = None
        self.indices = None

        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        #self.context_encoder.reset(num_tasks)

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        if len(r.shape) == 0:
            r = ptu.from_numpy(np.array([r])[None, None, ...])
        else:
            r = ptu.from_numpy(r[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)
 
    def update_context_dict(self, batch_dict, env):
        ''' append context dictionary containing single/multiple transitions to the current context '''
        o = ptu.from_numpy(batch_dict['observations'][None, ...])
        a = ptu.from_numpy(batch_dict['actions'][None, ...])
        next_o = ptu.from_numpy(batch_dict['next_observations'][None, ...])
        if callable(getattr(env, "sparsify_rewards", None)) and self.sparse_rewards:
            r = batch_dict['rewards']
            sr = []
            for r_entry in r:
                sr.append(env.sparsify_rewards(r_entry))
            r = ptu.from_numpy(np.array(sr)[None, ...])
        else:
            r = ptu.from_numpy(batch_dict['rewards'][None, ...])
        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, next_o], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), 0.05*ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def init_VQ(self):
        self.use_VQ = True
        self.VectorQuantizeCodebook.init_VQ(self.collected_context_embeddings)
        self.collected_context_embeddings = None

    def clear_collected_context_embeddings(self):
        self.collected_context_embeddings = None


    def collect_context_embeddings(self, context_embeddings):
        if self.collected_context_embeddings is None:
            self.collected_context_embeddings = context_embeddings
        else:
            self.collected_context_embeddings = torch.cat([self.collected_context_embeddings, context_embeddings], dim=0)

    def make_task_z_for_recon(self, context):
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)

        z_means = torch.mean(params, dim=1)
        # z_vars = torch.std(params, dim=1)
        return z_means

    def infer_posterior(self, context, task_indices=None, for_update=False):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        if task_indices is None:
            self.task_indices = np.zeros((context.size(0),))
        elif not hasattr(task_indices, '__iter__'):
            self.task_indices = np.array([task_indices])
        else:
            self.task_indices = np.array(task_indices)

        # self.params = params
        self.z_means = torch.mean(params, dim=1) # dim: task, batch, feature (latent dim)
        self.z_vars = torch.std(params, dim=1)
        # self.sample_z()
        self.sample_z(for_update)

    def encode_no_mean(self, context):
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        return params

    def encode_with_mean(self, context):
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        z_means = torch.mean(params, dim=1)
        return z_means

    def encode_for_match(self, context):
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        z_means = torch.mean(params, dim=1)
        z_vars = torch.std(params, dim=1)

        select_codes, indices, commitment_loss = self.VectorQuantizeCodebook(z_means)
        code_num = self.VectorQuantizeCodebook._codebook.embed[0].shape[0]
        task_num = context.size(0)
        match_dict = {}
        for i in range(task_num):
            if str(indices[i]) not in list(match_dict.keys()):
                match_dict[str(indices[i])] = []
                match_dict[str(indices[i])].append(i)
            else:
                match_dict[str(indices[i])].append(i)

        return z_means, match_dict

    def sample_z(self, for_update=False):
        if self.use_VQ:
            if for_update is True:
                CodeBook_before_match = self.VectorQuantizeCodebook._codebook.embed[0]

                select_codes, self.indices, commitment_loss = self.VectorQuantizeCodebook(self.z_means)
                entropies = entropy(self.VectorQuantizeCodebook.codebook, select_codes,
                                    self.VectorQuantizeCodebook.project_in(self.z_means))

                self.z = self.z_means
                self.code = select_codes

                self.commitment_loss = commitment_loss
                self.entropies = entropies
            else:
                select_codes = self.VectorQuantizeCodebook.forward_for_test(self.z_means)
                # z_means_score = self.context_discriminator(self.z_means)
                # select_codes_score = self.context_discriminator(select_codes)
                # self.z, self.z_means_score, self.select_codes_score = self.context_discriminator.compute_softmax_result(
                #     self.z_means, z_means_score,
                #     select_codes, select_codes_score
                # )
                self.z = self.z_means
                self.code = None
                self.commitment_loss = None
                self.entropies = None

        else:
            self.z = self.z_means

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context, use_VQ=False, task_indices=None, for_update=False):
        ''' given context, get statistics under the current policy of a set of observations '''
        # self.infer_posterior(context, task_indices=task_indices)
        # self.sample_z()

        if self.use_VQ is True and use_VQ is True:
            self.infer_posterior(context, task_indices=task_indices, for_update=for_update)
            # self.sample_z(for_update)
            # self.collect_context_embeddings(self.z)
        elif self.use_VQ is False and use_VQ is True:
            self.use_VQ = True
            self.init_VQ()
            self.infer_posterior(context, task_indices=task_indices, for_update=for_update)
            # self.sample_z(for_update)
        elif self.use_VQ is False and use_VQ is False:
            self.infer_posterior(context, task_indices=task_indices, for_update=for_update)
            # self.sample_z(for_update)
            # self.collect_context_embeddings(self.z)
        else:
            self.infer_posterior(context, task_indices=task_indices, for_update=for_update)
            # self.sample_z(for_update)

        task_z = self.z

        commitment_loss = self.commitment_loss
        entropies = self.entropies
        task_z_before_match = self.z_means

        # self.meta_batch * self.batch_size * dim(obs)
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        task_z_before_match = [z.repeat(b, 1) for z in task_z_before_match]
        task_z_before_match = torch.cat(task_z_before_match, dim=0)

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.policy(t, b, in_, reparameterize=True, return_log_prob=True)

        task_z_vars = [z.repeat(b, 1) for z in self.z_vars]
        task_z_vars = torch.cat(task_z_vars, dim=0)
        if not for_update:
            return policy_outputs, task_z, task_z_vars
        else:
            return policy_outputs, task_z, task_z_vars, commitment_loss, entropies, \
                   self.code, self.indices

    def log_diagnostics(self, eval_statistics):
        # adds logging data about encodings to eval_statistics
        # z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        

        for i in range(len(self.z_means[0])):
            z_mean = ptu.get_numpy(self.z_means[0][i])
            name = 'Z mean eval' + str(i)
            eval_statistics[name] = z_mean
        #z_mean1 = ptu.get_numpy(self.z_means[0][0])
        #z_mean2 = ptu.get_numpy(self.z_means[0][1])
        #z_mean3 = ptu.get_numpy(self.z_means[0][2])
        #z_mean4 = ptu.get_numpy(self.z_means[0][3])
        #z_mean5 = ptu.get_numpy(self.z_means[0][4])

        #eval_statistics['Z mean eval1'] = z_mean1
        #eval_statistics['Z mean eval2'] = z_mean2
        #eval_statistics['Z mean eval3'] = z_mean3
        #eval_statistics['Z mean eval4'] = z_mean4
        #eval_statistics['Z mean eval5'] = z_mean5
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z variance eval'] = z_sig

        # eval_statistics['Z mean eval'] = z_mean
        # eval_statistics['Z variance eval'] = z_sig
        eval_statistics['task_idx'] = self.task_indices[0]

    @property
    def networks(self):
        # return [self.context_encoder, self.policy]
        return [self.context_encoder, self.policy, self.VectorQuantizeCodebook]