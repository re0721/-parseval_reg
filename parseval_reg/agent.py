#####
# PPO/RPO implementation from CleanRL  (https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy)
# 
#####


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from collections import OrderedDict

from utils import GroupSort
from utils import (create_block_diag_matrix, IdentityActivation,
                                    DiagLinear, ScaleLayer, ConcatReLU)

import geoopt
from geoopt.manifolds import Stiefel
from geoopt.tensor import ManifoldParameter

from pion_optimizer import Pion, PionMinimal
from poet_official import PoetOfficialLinear


# torch.manual_seed(5)
# import random
# random.seed(5)
# np.random.seed(5)


class PPO_Agent:
    def __init__(self, env, device, learning_rate=0.0003, num_envs=1, rollout_num_steps=128, 
                 gamma=0.99, gae_lambda=0.95, num_minibatches=4, update_epochs=4, minibatch_size=32,
                 norm_adv=True, clip_coef=0.2, clip_vloss=True, ent_coef=0.0,
                 vf_coef=0.5, max_grad_norm=0.5, target_kl=None,
                 layer_norm=False, layer_norm_no_params=False,
                 weight_decay=0, tuned_adam=False,
                 network_type='mlp',
                 weight_init='orthogonal', add_diag_layer=False,
                 parseval_reg=0, parseval_norm=False, parseval_last_layer=False, parseval_num_groups=1,
                 perturb=0.0, perturb_dist='xavier',
                 regen=0.0, regen_wasserstein=False,
                 iso_reg=0.0, spectral_reg=0.0,
                 l2_er_weight_decay=0.0, l2_er_beta=0.0,
                 rpo_alpha=0, net_width=64, net_activation='tanh', init_gain=None,
                 input_scale=1, learnable_input_scale=False, lie_group=False, pion=False,
                 pion_block_size=None, oft=False, poet=False, oet_num_blocks=1,
                 poet_exact_cayley=False,
                 pion_multiplicative=False, pion_minimal=False, pion_minimal_rms=False,
                 seed=None, *args, **kwargs):

        self.env = env
        self.device = device
        self.learning_rate = learning_rate
        self.num_envs = num_envs
        self.rollout_num_steps = rollout_num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.seed = seed

        self.layer_norm = layer_norm
        self.layer_norm_no_params = layer_norm_no_params
        self.weight_decay = weight_decay
        self.tuned_adam = tuned_adam
        self.parseval_reg = parseval_reg
        self.parseval_norm = parseval_norm
        # self.parseval_last_layer = parseval_last_layer  # removed from uses
        self.parseval_num_groups = parseval_num_groups
        self.rpo_alpha = rpo_alpha
        self.lie_group = lie_group
        self.pion = pion
        self.pion_block_size = pion_block_size
        self.pion_multiplicative = pion_multiplicative
        self.pion_minimal = pion_minimal
        self.pion_minimal_rms = pion_minimal_rms
        self.oft = oft
        self.poet = poet
        self.oet_num_blocks = oet_num_blocks
        self.poet_exact_cayley = poet_exact_cayley

        self.perturb = perturb  # for shrink-and-perturb. Shrink is implemented through weight decay
        self.perturb_dist = perturb_dist  # either "xavier" or "orthogonal". The type of noise to add.
        self.regen = regen  # renegerative regularizer coefficient. l2-reg towards the initial parameters
        self.regen_wasserstein = regen_wasserstein
        self.iso_reg = iso_reg  # dynamical-isometry (Gram-deviation) regularizer coefficient
        self.spectral_reg = spectral_reg  # spectral-norm regularizer coefficient (sigma_max -> 1)
        self.l2_er_weight_decay = l2_er_weight_decay  # L2-ER: L2 weight decay coefficient
        self.l2_er_beta = l2_er_beta  # L2-ER: effective-rank penalty coefficient

        self.network_type = network_type
        self.add_diag_layer = add_diag_layer
        self.weight_init = weight_init
        self.net_width = net_width
        self.net_activation = net_activation
        self.init_gain = init_gain  # gain value for orthogonal init
        self.input_scale = input_scale
        self.learnable_input_scale = learnable_input_scale

        self.discrete_action_space = isinstance(self.env.env.action_space, gym.spaces.Discrete)  # check to dynamically make network for continuous or discrete action spaces

        if seed is not None:
            torch.manual_seed(seed)

        if minibatch_size is not None:
            self.minibatch_size = minibatch_size
            self.rollout_num_steps = self.minibatch_size * self.num_minibatches
            # if minibatch size is specified, we override the rollout_num_steps with appropriate value
            self.batch_size = int(self.num_envs * self.rollout_num_steps)
        else:
            self.batch_size = int(self.num_envs * self.rollout_num_steps)
            self.minibatch_size = int(self.batch_size // self.num_minibatches)

        self.update_counter = 0
        self.step = 0

        self.mode = 'train'
        self.image_obs = len(self.env.env.observation_space.shape) > 1

        self.agent_networks = AgentNetworks(env.env, network_type, layer_norm=self.layer_norm, layer_norm_no_params=self.layer_norm_no_params,
                                            rpo_alpha=self.rpo_alpha, weight_init=self.weight_init, init_gain=self.init_gain,
                                            net_width=self.net_width, activation=self.net_activation,
                                            parseval_reg=self.parseval_reg, add_diag_layer=self.add_diag_layer,
                                            input_scale=input_scale, learnable_input_scale=learnable_input_scale,
                                            lie_group=self.lie_group,
                                            oft=self.oft, poet=self.poet, oet_num_blocks=self.oet_num_blocks,
                                            poet_exact_cayley=self.poet_exact_cayley,
                                            discrete_action_space=self.discrete_action_space)
        if self.regen > 0:  # make a copy of the initial weights (we don't update these)
            self.agent_networks_init = AgentNetworks(env.env, network_type, layer_norm=self.layer_norm, layer_norm_no_params=self.layer_norm_no_params,
                                            rpo_alpha=self.rpo_alpha, weight_init=self.weight_init, init_gain=self.init_gain,
                                            net_width=self.net_width, activation=self.net_activation, add_diag_layer=self.add_diag_layer,
                                            parseval_reg=self.parseval_reg, input_scale=input_scale, learnable_input_scale=learnable_input_scale,
                                            discrete_action_space=self.discrete_action_space)
            self.agent_networks_init.load_state_dict(self.agent_networks.state_dict())
            self.agent_networks_init.requires_grad_(False)

        if self.pion:
            # Spectrum-preserving Lie-group optimizer (Pion).
            # multiplicative variant keeps the paper-Algorithm-1 cross term (beta2=0.95);
            # official additive update uses beta2=0.999.
            if self.pion_multiplicative:
                self.optimizer = Pion(self.agent_networks.parameters(), lr=self.learning_rate,
                                      betas=(0.9, 0.95), multiplicative=True)
            else:
                self.optimizer = Pion(self.agent_networks.parameters(), lr=self.learning_rate)
        elif self.pion_minimal:
            # Minimal Pion: orthogonal-transformation core only (no momentum/RMS/additive).
            self.optimizer = PionMinimal(self.agent_networks.parameters(), lr=self.learning_rate)
        elif self.pion_minimal_rms:
            # Minimal Pion + RMS scaling (still no momentum, multiplicative).
            self.optimizer = PionMinimal(self.agent_networks.parameters(), lr=self.learning_rate, rms_scaling=True)
        elif self.lie_group:
            # StiefelLinear uses ManifoldParameter -> RiemannianAdam.
            self.optimizer = geoopt.optim.RiemannianAdam(self.agent_networks.parameters(), lr=self.learning_rate, eps=1e-5)
        elif self.tuned_adam:
            if weight_decay > 0:
                self.optimizer = optim.AdamW(self.agent_networks.parameters(), lr=self.learning_rate,
                                             betas=(0.9, 0.9), eps=1e-3, weight_decay=weight_decay)
            else:
                self.optimizer = optim.Adam(self.agent_networks.parameters(), lr=self.learning_rate,
                                            betas=(0.9, 0.9), eps=1e-3)
        else:
            # Decoupled weight decay (AdamW): combine the base weight_decay with L2-ER's.
            # Official L2-ER applies weight decay in the optimizer (optax.add_decayed_weights),
            # NOT as an L2 term in the loss.
            wd = weight_decay + l2_er_weight_decay
            if wd > 0:
                self.optimizer = optim.AdamW(self.agent_networks.parameters(), lr=self.learning_rate,
                                             eps=1e-5, weight_decay=wd)
            else:
                self.optimizer = optim.Adam(self.agent_networks.parameters(), lr=self.learning_rate, eps=1e-5)

        self.obs = torch.zeros((self.rollout_num_steps, self.num_envs, np.array(env.env.observation_space.shape).prod())).to(device)
        self.actions = torch.zeros((self.rollout_num_steps, self.num_envs) + self.env.env.action_space.shape).to(device)
        self.logprobs = torch.zeros((self.rollout_num_steps, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.rollout_num_steps, self.num_envs)).to(device)
        self.dones = torch.zeros((self.rollout_num_steps, self.num_envs)).to(device)
        self.values = torch.zeros((self.rollout_num_steps, self.num_envs)).to(device)

        # for logs
        self.explained_var = None
        self.entropy = None

        # print("PARAMETERS")
        # for name, param in self.agent_networks.actor_mean.named_parameters():
        #     print(name, param.shape)

    def act(self, obs):
        if self.image_obs:
            obs = obs.flatten()
        if self.mode == 'train':
            with torch.no_grad():
                action, logprob, _, value = self.agent_networks.get_action_and_value(torch.FloatTensor(obs).to(self.device))
                # this stores the relevant quantities to the update after
                self.values[self.step] = value.flatten()
                self.actions[self.step] = action
                self.logprobs[self.step] = logprob

        elif self.mode == 'eval':
            with torch.no_grad():
                action, logprob, _, value = self.agent_networks.get_action_and_value(torch.FloatTensor(obs).to(self.device))

        return action.cpu().numpy()

    def update(self, obs, next_obs, action, reward, terminated, truncated, info):
        if self.image_obs:
            obs = obs.flatten()
            next_obs = next_obs.flatten()

        ## update saved transition info'
        self.rewards[self.step] = torch.tensor(reward).to(self.device).view(-1)
        obs = torch.Tensor(obs).to(self.device)
        if self.discrete_action_space:
            next_done = terminated# or truncated
        else:
            next_done = terminated or truncated
        next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor([next_done]).to(self.device)
        # edited done from original CleanRL code. self.dones[t] here corresponds to dones[t+1] in original code

        self.obs[self.step] = obs  # todo check
        self.dones[self.step] = next_done  # terminated

        self.step += 1
        self.step = self.step % self.rollout_num_steps

        # do an update if enough transitions have been collected
        if self.step == 0:
            self._update_parameters(next_obs, next_done)
            # print('updating parameters')
        self.update_counter += 1

    def _update_parameters(self, next_obs, next_done):
        ## don't do the following, did in act()
        # with torch.no_grad():
        #     action, logprob, _, value = agent.get_action_and_value(next_obs)
        #     values[step] = value.flatten()
        # actions[step] = action
        # logprobs[step] = logprob
        ##


        ## compute GAE targets
        with torch.no_grad():
            next_value = self.agent_networks.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.rollout_num_steps)):
                if t == self.rollout_num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        ## optimize policy and value network
        b_obs = self.obs.reshape((-1, np.array(self.env.env.observation_space.shape).prod()))
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.env.env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                if self.discrete_action_space:
                    _, newlogprob, entropy, newvalue = self.agent_networks.get_action_and_value(b_obs[mb_inds],
                                                                                                b_actions.long()[mb_inds])
                else:
                    _, newlogprob, entropy, newvalue = self.agent_networks.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                ##### Parseval reg
                if self.parseval_reg > 0:
                    def parseval_reg_network(named_parameters):
                        loss_reg = 0

                        for name, param in named_parameters:
                            if 'weight' in name and 'orthog' in name and param.requires_grad:
                                # print(self.init_gain)
                                if self.init_gain is None:
                                    scale = 2 # sqrt(2)**2
                                else:
                                    scale = self.init_gain **2

                                if self.parseval_norm:
                                    temp_par = param / torch.norm(param, dim=1).view(-1,1)
                                else:
                                    temp_par = param

                                # weight matrices right multiply by their inputs
                                if self.parseval_num_groups == 1:
                                    loss_reg = loss_reg + torch.norm(
                                        torch.matmul(temp_par, temp_par.t()) - scale * torch.eye(temp_par.shape[0]),
                                        p='fro') ** 2
                                elif self.parseval_num_groups > 1:
                                    if self.net_width % self.parseval_num_groups != 0:
                                        raise AssertionError(
                                            f'net_width ({self.net_width}) has to be divisible by parseval_num_groups ({self.parseval_num_groups})')

                                    neuron_group_size = self.net_width // self.parseval_num_groups
                                    mask_matrix = create_block_diag_matrix(neuron_group_size, self.parseval_num_groups)

                                    loss_reg = loss_reg + torch.norm(
                                        mask_matrix * torch.matmul(temp_par, temp_par.t()) - scale * torch.eye(
                                            temp_par.shape[0]),
                                        p='fro') ** 2
                        return loss_reg

                    loss = loss + self.parseval_reg * (64 / self.net_width)**2 * parseval_reg_network(self.agent_networks.actor_mean.named_parameters())
                    loss = loss + self.parseval_reg * (64 / self.net_width)**2 * parseval_reg_network(
                        self.agent_networks.critic.named_parameters())

                    #  (self.net_width / 64) This factor is used since the original parseval_reg was tuned for width 64
                #####

                ##### ISO (dynamical isometry / Gram-deviation) regularizer
                if self.iso_reg > 0:
                    def iso_reg_network(named_parameters):
                        loss_iso = 0
                        for name, param in named_parameters:
                            if 'weight' in name and 'linear' in name and param.requires_grad:
                                W = torch.atleast_2d(param)
                                d_out, d_in = W.shape
                                if d_out >= d_in:
                                    gram = W.t() @ W
                                    loss_iso = loss_iso + torch.norm(
                                        gram - torch.eye(d_in, device=W.device, dtype=W.dtype), p='fro') ** 2
                                else:
                                    gram = W @ W.t()
                                    loss_iso = loss_iso + torch.norm(
                                        gram - torch.eye(d_out, device=W.device, dtype=W.dtype), p='fro') ** 2
                        return loss_iso

                    loss = loss + self.iso_reg * iso_reg_network(self.agent_networks.actor_mean.named_parameters())
                    loss = loss + self.iso_reg * iso_reg_network(self.agent_networks.critic.named_parameters())
                #####

                ##### Spectral (spectral-norm -> 1) regularizer, k=2, power iteration
                if self.spectral_reg > 0:
                    def spectral_reg_network(named_parameters):
                        loss_spec = 0
                        for name, param in named_parameters:
                            if 'weight' in name and 'linear' in name and param.requires_grad:
                                W = torch.atleast_2d(param)
                                # single power iteration to estimate sigma_max^2
                                u = torch.randn(W.shape[1], device=W.device, dtype=W.dtype)
                                u = u / torch.norm(u)
                                sigma_max_sq = torch.norm(W.t() @ (W @ u))
                                loss_spec = loss_spec + (sigma_max_sq - 1.0) ** 2
                            elif 'bias' in name and param.requires_grad:
                                loss_spec = loss_spec + torch.norm(param) ** 4  # ||b||_2^{2k}, k=2
                        return loss_spec

                    loss = loss + self.spectral_reg * spectral_reg_network(self.agent_networks.actor_mean.named_parameters())
                    loss = loss + self.spectral_reg * spectral_reg_network(self.agent_networks.critic.named_parameters())
                #####

                ##### L2-ER: decoupled update applied after optimizer.step() below #####

                if self.regen > 0:  # add regenerative regularization
                    loss_regen = 0
                    # two versions: standard or wasserstein (sorted parameters)
                    if not self.regen_wasserstein:
                        # standard
                        for param, param_init in zip(self.agent_networks.actor_mean.parameters(),
                                                     self.agent_networks_init.actor_mean.parameters()):
                            loss_regen = loss_regen + torch.linalg.vector_norm(param - param_init)**2 / param.numel()  # l2-norm squared
                        for param, param_init in zip(self.agent_networks.critic.parameters(),
                                                     self.agent_networks_init.critic.parameters()):
                            loss_regen = loss_regen + torch.linalg.vector_norm(param - param_init)**2 / param.numel()

                    else:
                        # wasserstein version: for each layer, sort params after flattening
                        # a little unclear how to do this with layer norm or the bias units
                        # well, I'm doing it the straightforward way and just applying it directly
                        def compute_regen_wasserstein(parameters, parameters_init):
                            loss_regen_wass = 0
                            for param, param_init in zip(parameters, parameters_init):
                                sorted_param, _ = param.view(-1).sort()
                                sorted_param_init, _ = param_init.view(-1).sort()
                                loss_regen_wass = loss_regen_wass + torch.linalg.vector_norm(
                                    sorted_param - sorted_param_init) ** 2 / param.numel()

                            return loss_regen_wass

                        loss_regen = loss_regen + compute_regen_wasserstein(self.agent_networks.actor_mean.parameters(),
                                                                            self.agent_networks_init.actor_mean.parameters())
                        loss_regen = loss_regen + compute_regen_wasserstein(self.agent_networks.critic.parameters(),
                                                                            self.agent_networks_init.critic.parameters())
                    loss = loss + self.regen * loss_regen


                #####

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent_networks.parameters(), self.max_grad_norm)
                self.optimizer.step()

                ##### L2-ER effective-rank: decoupled plain gradient ascent (matches official rlopt)
                # Official L2-ER applies the effective-rank penalty as a SEPARATE gradient step
                # theta += er_lr * grad(erank), decoupled from the Adam task update (like AdamW
                # decouples weight decay). The L2 weight decay is likewise decoupled via AdamW above.
                if self.l2_er_beta > 0:
                    def erank(A):
                        try:
                            s = torch.linalg.svdvals(A)
                        except torch._C._LinAlgError:
                            # gesdd 对"太多重复奇异值"不收敛（ER 会把谱推向均匀）。
                            # 加微小噪声打破对称后重试；噪声无梯度，梯度仍只流过 A。
                            s = torch.linalg.svdvals(A + 1e-6 * torch.randn_like(A))
                        s = torch.abs(s)
                        total = s.sum().clamp_min(1e-8)
                        p = s / total
                        H = -(p * torch.log(p + 1e-12)).sum()
                        return torch.exp(H)

                    def hidden_activations(net, obs):
                        x = obs
                        acts = []
                        for module in net:
                            if isinstance(module, nn.Linear):
                                acts.append(x)
                            x = module(x)
                        return acts[1:]

                    loss_er = 0
                    for net in (self.agent_networks.actor_mean, self.agent_networks.critic):
                        for A in hidden_activations(net, b_obs[mb_inds]):
                            loss_er = loss_er + erank(A)

                    # Save the task grads (get_log_quantities() reads param.grad after the update).
                    task_grads = {n: p.grad.clone() for n, p in self.agent_networks.named_parameters() if p.grad is not None}
                    self.optimizer.zero_grad()
                    loss_er.backward()
                    with torch.no_grad():
                        for param in self.agent_networks.parameters():
                            if param.grad is not None:
                                param.add_(self.l2_er_beta * param.grad)  # theta += beta * grad(erank)
                    self.optimizer.zero_grad()
                    for n, p in self.agent_networks.named_parameters():
                        if n in task_grads:
                            p.grad = task_grads[n]
                #####

                ### Add noise for shrink-and-perturb
                # make a new model with same weights
                if self.perturb > 0:
                    dummy_model = AgentNetworks(self.env.env, layer_norm=self.layer_norm,
                                                layer_norm_no_params=self.layer_norm_no_params,
                                                tsallis_ent_coef=None,
                                                weight_init=self.perturb_dist, init_gain=self.init_gain,
                                                net_width=self.net_width,
                                                activation=self.net_activation,
                                                network_type=self.network_type,
                                                add_diag_layer=self.add_diag_layer,
                                                input_scale=self.input_scale,
                                                discrete_action_space=self.discrete_action_space)
                                                # note weight_init can be different

                    # iterate over params and add noise
                    # actor
                    for name_param, dummy_name_param in zip(self.agent_networks.actor_mean.named_parameters(), dummy_model.actor_mean.named_parameters()):
                        name, param = name_param

                        dummy_name, dummy_param = dummy_name_param
                        if 'weight' in name and 'linear' in name:  # we only update the linear layers (not diagonal ones)
                            # print('perturbing!!!')
                            if param.requires_grad:
                                with torch.no_grad():
                                    param.data = param.data + self.perturb * dummy_param.data
                    # critic
                    for name_param, dummy_name_param in zip(self.agent_networks.critic.named_parameters(), dummy_model.critic.named_parameters()):
                        name, param = name_param
                        dummy_name, dummy_param = dummy_name_param
                        if 'weight' in name and 'linear' in name:  # we only update the linear layers (not diagonal ones)
                            if param.requires_grad:
                                with torch.no_grad():
                                    param.data = param.data + self.perturb * dummy_param.data

            if self.target_kl is not None:
                if approx_kl > self.target_kl:
                    break


        ### for logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        self.explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y  # evaluates the critic
        self.entropy = entropy_loss.clone().item()

        # gradients for actor without considering entropy?

    def save_model(self, save_file_path):
        torch.save(self.agent_networks.state_dict(), save_file_path)

    def load_model(self, save_file_path):
        ''' Loads the model from saved weights '''
        if self.device == torch.device('cpu'):
            self.agent_networks.load_state_dict(torch.load(save_file_path, map_location=torch.device('cpu')))
        else:
            self.agent_networks.load_state_dict(torch.load(save_file_path))

    def eval(self):
        self.mode = 'eval'

    def train(self):
        self.mode = 'train'

    def get_log_quantities(self):
        logged_values = {}
        extra_layers = self.layer_norm
        with torch.no_grad():
            ### Save states for testing later

            ### gradient norm
            actor_grads = []
            for name, param in self.agent_networks.actor_mean.named_parameters():
                if param.requires_grad:
                    actor_grads.append(param.grad.view(-1))
            actor_grads = torch.cat(actor_grads)
            actor_norm = torch.linalg.vector_norm(actor_grads, ord=2)

            critic_grads = []
            for name, param in self.agent_networks.critic.named_parameters():
                if param.requires_grad:
                    critic_grads.append(param.grad.view(-1))
            critic_grads = torch.cat(critic_grads)
            critic_norm = torch.linalg.vector_norm(critic_grads, ord=2)

            logged_values['actor_grad_norm'] = actor_norm.item()
            logged_values['critic_grad_norm'] = critic_norm.item()

            ### params and gradients singular values
            actor_gradient_singular_values = []
            actor_singular_values = []
            for name, param in self.agent_networks.actor_mean.named_parameters():
                if 'weight' in name and 'linear' in name and param.grad is not None:
                    # print("SHAPE", torch.atleast_2d(param.grad).shape)
                    _, grad_singular_values, _ = torch.svd(torch.atleast_2d(param.grad))
                    _, singular_values, _ = torch.svd(torch.atleast_2d(param))

                    actor_gradient_singular_values.append(grad_singular_values.cpu().numpy())
                    actor_singular_values.append(singular_values.cpu().numpy())

            critic_gradient_singular_values = []
            critic_singular_values = []
            for name, param in self.agent_networks.critic.named_parameters():
                if 'weight' in name and 'linear' in name and param.grad is not None:
                    _, grad_singular_values, _ = torch.svd(torch.atleast_2d(param.grad))
                    _, singular_values, _ = torch.svd(torch.atleast_2d(param))

                    critic_gradient_singular_values.append(grad_singular_values.cpu().numpy())
                    critic_singular_values.append(singular_values.cpu().numpy())

            logged_values['actor_grad_singular_values'] = actor_gradient_singular_values
            logged_values['actor_param_singular_values'] = actor_singular_values
            logged_values['critic_grad_singular_values'] = critic_gradient_singular_values
            logged_values['critic_param_singular_values'] = critic_singular_values

            ### parameter and bias norms
            actor_weight_norms = []
            actor_bias_norms = []
            for name, param in self.agent_networks.actor_mean.named_parameters():
                # print("ACTOR", name)
                if ('weight' in name or 'bias' in name) and param.requires_grad:
                    # print("ACTOR2", name)
                    if 'weight' in name:
                        param = torch.atleast_2d(param)
                        weight_norm = torch.linalg.matrix_norm(param, ord='fro')
                        actor_weight_norms.append(weight_norm.item())
                    elif 'bias' in name:
                        bias_norm = torch.linalg.vector_norm(param, ord=2)
                        actor_bias_norms.append(bias_norm.item())

            critic_weight_norms = []
            critic_bias_norms = []
            for name, param in self.agent_networks.critic.named_parameters():
                # print("ACTOR", name)
                if ('weight' in name or 'bias' in name) and param.requires_grad:
                    # print("ACTOR2", name)
                    if 'weight' in name:
                        param = torch.atleast_2d(param)
                        weight_norm = torch.linalg.matrix_norm(param, ord='fro')
                        critic_weight_norms.append(weight_norm.item())
                    elif 'bias' in name:
                        bias_norm = torch.linalg.vector_norm(param, ord=2)
                        critic_bias_norms.append(bias_norm.item())

            logged_values['actor_weight_norms'] = actor_weight_norms
            logged_values['actor_bias_norms'] = actor_bias_norms
            logged_values['critic_weight_norms'] = critic_weight_norms
            logged_values['critic_bias_norms'] = critic_bias_norms

            ### stable rank
            # compute stable rank, ratio of squared frobenius norm to squared spectral norm
            actor_stable_ranks = []
            for name, param in self.agent_networks.actor_mean.named_parameters():
                # print("ACTOR", name)

                if 'weight' in name and 'linear' in name:
                    param = torch.atleast_2d(param)
                    stable_rank = (torch.linalg.matrix_norm(param, ord='fro') / torch.linalg.matrix_norm(param, ord=2)) **2
                    actor_stable_ranks.append(stable_rank.item())

            critic_stable_ranks = []
            for name, param in self.agent_networks.critic.named_parameters():
                if 'weight' in name and 'linear' in name:
                    param = torch.atleast_2d(param)
                    stable_rank = (torch.linalg.matrix_norm(param, ord='fro') / torch.linalg.matrix_norm(param, ord=2)) **2
                    critic_stable_ranks.append(stable_rank.item())

            logged_values['actor_matrix_stable_rank'] = actor_stable_ranks
            logged_values['critic_matrix_stable_rank'] = critic_stable_ranks

            ### cosine similarity
            actor_cosine_sim_per_layer = []
            for name, param in self.agent_networks.actor_mean.named_parameters():
                if 'weight' in name and 'linear' in name:
                    # only consider the angle between vectors
                    # we normalize the weights row-wise and then regularize towards identity
                    normed_param = torch.nn.functional.normalize(param, dim=1)
                    cosine_sim = torch.norm(
                        torch.matmul(normed_param, normed_param.t()) - torch.eye(param.shape[0]),
                        p='fro') ** 2  # removed the diagonal entries

                    cosine_sim = cosine_sim / (param.shape[0]**2 - param.shape[0])  # avg over entries
                    actor_cosine_sim_per_layer.append(cosine_sim.item())

            critic_cosine_sim_per_layer = []
            for name, param in self.agent_networks.critic.named_parameters():
                if 'weight' in name and 'linear' in name:
                    # only consider the angle between vectors
                    # we normalize the weights row-wise and then regularize towards identity
                    normed_param = torch.nn.functional.normalize(param, dim=1)
                    cosine_sim = torch.norm(
                        torch.matmul(normed_param, normed_param.t()) - torch.eye(param.shape[0]),
                        p='fro') ** 2  # removed the diagonal entries
                    if (param.shape[0] ** 2 - param.shape[0]) != 0:
                        cosine_sim = cosine_sim / (param.shape[0]**2 - param.shape[0])  # avg over entries
                    critic_cosine_sim_per_layer.append(cosine_sim.item())
            logged_values['actor_cosine_sim'] = actor_cosine_sim_per_layer
            logged_values['critic_cosine_sim'] = critic_cosine_sim_per_layer

        return logged_values


class StiefelLinear(nn.Module):
    """Dense layer whose weight lives on the Stiefel manifold (orthonormal rows,
    W W^T = I). This is Lie-group / Riemannian optimization of orthogonal weights
    — the 'hard' counterpart of Parseval's soft regularization.

    geoopt's Stiefel constrains orthonormal COLUMNS (X^T X = I, X in R^{n x m},
    n >= m), so we store X = W^T (shape [in, out]) on the manifold, which makes
    W W^T = I (orthonormal rows of the linear weight), matching Parseval.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert in_features >= out_features, f"Stiefel (orthonormal rows) needs in>=out, got {in_features}<{out_features}"
        self.manifold = Stiefel(canonical=True)  # X^T X = I, X shape [in, out]
        X = torch.empty(in_features, out_features)
        torch.nn.init.orthogonal_(X, gain=1.0)  # orthonormal columns of X -> W W^T = I
        self.weight = ManifoldParameter(X, manifold=self.manifold)  # stores W^T
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.t(), self.bias)


class OETLinear(nn.Module):
    """Orthogonal Equivalence Transformation linear layer (OFT / POET).

    Follows the official implementation (Zeju1997/oft): the orthogonal matrix is
    parameterized by the CAYLEY transform R = (I + skew)(I - skew)^{-1}, where
    skew = 0.5*(Q - Q^T) and Q is a learnable matrix initialized to zero (so R
    starts at identity). R is block-diagonal (num_blocks blocks) to avoid the
    O(d^3) inverse for wide input layers.

    OFT  : W = W0 @ R^T        (one-sided, input rotation)
    POET : W = P @ W0 @ R^T    (two-sided, P on output, R on input)
    """
    def __init__(self, in_features, out_features, two_sided=False, bias=True, num_blocks=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.two_sided = two_sided
        # fall back to full R if num_blocks doesn't divide in_features
        self.num_blocks = num_blocks if in_features % num_blocks == 0 else 1
        self.block_size = in_features // self.num_blocks
        # frozen base weight: Gaussian init -> non-degenerate spectrum
        self.W0 = nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.xavier_normal_(self.W0, gain=1.0)
        self.W0.requires_grad_(False)
        # learnable Q (skew-symmetric via Cayley), init zero -> R = I
        self.Q_in = nn.Parameter(torch.zeros(self.num_blocks, self.block_size, self.block_size))
        if two_sided:
            self.Q_out = nn.Parameter(torch.zeros(out_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def _cayley_blocks(self, Q):
        # Q: (nb, b, b) -> (nb, b, b) orthogonal blocks via Cayley
        skew = 0.5 * (Q - Q.transpose(1, 2))
        I = torch.eye(self.block_size, device=Q.device, dtype=Q.dtype).unsqueeze(0)
        return (I + skew) @ torch.inverse(I - skew)

    def _cayley(self, Q):
        # Q: (d, d) -> (d, d) orthogonal via Cayley
        skew = 0.5 * (Q - Q.t())
        I = torch.eye(Q.shape[0], device=Q.device, dtype=Q.dtype)
        return (I + skew) @ torch.inverse(I - skew)

    def forward(self, x):
        R_blocks = self._cayley_blocks(self.Q_in)  # (nb, b, b)
        # W = W0 @ R^T, block-wise on the input side
        W0_blocks = self.W0.view(self.out_features, self.num_blocks, self.block_size)
        W = torch.einsum('onb,nbs->ons', W0_blocks, R_blocks.transpose(1, 2))
        W = W.reshape(self.out_features, self.in_features)
        if self.two_sided:
            P = self._cayley(self.Q_out)  # (dout, dout) orthogonal
            W = P @ W
        return torch.nn.functional.linear(x, W, self.bias)


def orthogonal_layer_init(layer, gain=np.sqrt(2), bias_const=0.0):
    if gain is None:
        gain = np.sqrt(2)
    torch.nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def xavier_layer_init(layer, gain=1):
    if gain is None:
        gain = 1
    torch.nn.init.xavier_normal_(layer.weight, gain)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, 0.0)
    return layer


def ones_layer_init(layer, gain=1.0, bias_const=0.0):
    """All-constant (rank-1) init: the degenerate extreme of the spectrum.

    A constant matrix has a single nonzero singular value (stable rank = 1).
    For spectrum-preserving methods this LOCKS the spectrum at rank 1, so the
    network keeps exactly one effective direction and cannot learn. It is the
    negative control for 'the preserved spectrum must be a healthy, shaped one'.
    `gain` scales the constant (gain=1 -> all ones, gain=0.1 -> all 0.1).
    """
    if gain is None:
        gain = 1.0
    torch.nn.init.constant_(layer.weight, gain)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def standard_layer_init(layer, gain=None, std=0.1, bias_const=0.0):
    """Fixed-variance zero-mean Gaussian init (Hinton & Salakhutdinov 2006, N(0, 0.01)).

    Unlike xavier (variance scaled by layer dims), this uses a FIXED variance
    0.01 (std = sqrt(0.01) = 0.1) for every layer, so the spectrum scale is
    layer-independent. The spectrum is shaped but the scale does not depend on
    layer width.
    """
    if gain is not None:
        std = std * gain
    torch.nn.init.normal_(layer.weight, mean=0.0, std=std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def lecun_layer_init(layer, gain=None, bias_const=0.0):
    """LeCun uniform init: U(-sqrt(3/fan_in), sqrt(3/fan_in)), variance 1/fan_in.

    This is the init used by the official L2-ER (Spectral Collapse) RL code
    (rlopt/models.py, `jax.nn.initializers.lecun_uniform`). `gain` is ignored so
    that EVERY layer (including the output head) gets plain LeCun uniform,
    matching the paper's RL code exactly. Bias is set to a constant.
    """
    torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='linear')
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def identity_layer_init(layer, gain=None, bias_const=0.0):
    """Identity init: W = I (orthogonal, all singular values = 1).

    For non-square matrices, torch.nn.init.eye_ fills the min(d_in, d_out)
    leading diagonal with 1 and leaves the rest 0, so the matrix is a partial
    isometry (all nonzero singular values = 1). If `gain` is given, W = gain * I.
    """
    torch.nn.init.eye_(layer.weight)
    if gain is not None:
        with torch.no_grad():
            layer.weight.mul_(gain)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentNetworks(nn.Module):
    def __init__(self, env, network_type, weight_init='orthogonal', init_gain=None,
                 layer_norm=False, layer_norm_no_params=False, tsallis_ent_coef=None, rpo_alpha=0,
                 net_width=64, activation=None, parseval_reg=0, add_diag_layer=False,
                 input_scale=1, learnable_input_scale=False, lie_group=False, oft=False, poet=False,
                 oet_num_blocks=1, poet_exact_cayley=False, project_weight_dim=None, discrete_action_space=None):
        super().__init__()
        self.network_type = network_type  # mlp or resnet
        self.tsallis_ent_coef = tsallis_ent_coef
        self.rpo_alpha = rpo_alpha
        self.weight_init = weight_init
        self.init_gain = init_gain
        self.net_width = net_width
        self.activation = activation
        self.parseval_reg = parseval_reg
        self.add_diag_layer = add_diag_layer
        self.lie_group = lie_group
        self.oft = oft
        self.poet = poet
        self.oet_num_blocks = oet_num_blocks
        self.poet_exact_cayley = poet_exact_cayley
        self.input_scale = input_scale
        self.learnable_input_scale = learnable_input_scale
        self.project_weight_dim = project_weight_dim

        self.discrete_action_space = discrete_action_space

        # for k, v in locals().items():
        #     print(k, v)


        num_hidden = net_width

        if network_type.lower() == 'mlp':
            self.actor_mean, self.critic = self.build_network(env, num_hidden, layer_norm, layer_norm_no_params,
                                                              add_diag_layer, activation,
                                                              weight_init, init_gain, parseval_reg, input_scale,
                                                              learnable_input_scale, lie_group, oft, poet,
                                                              oet_num_blocks, poet_exact_cayley, project_weight_dim,
                                                              discrete_action_space)
            
        output_size = env.action_space.n if discrete_action_space else np.prod(env.action_space.shape)
        self.actor_logstd = nn.Parameter(torch.zeros(1, output_size))

        # print("DISCRETE ACTIONS:", discrete_action_space)
        if discrete_action_space:  #  use the appropriate function to get actions
            self.get_action_and_value = self._get_action_and_value_discrete
        else:
            self.get_action_and_value = self._get_action_and_value_continuous

    def get_value(self, x):
        return self.critic(x)


    def _get_action_and_value_continuous(self, x, action=None):
        # hmm add tsallis entropy?
        x = torch.atleast_2d(x)   # adds a batch dimension if there's only one

        action_mean = self.actor_mean(x)
        # print('x', x.shape)
        # print('action_mean', action_mean.shape)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
            action = action.squeeze()  # not using vector env
            # print(action)
        else:
            if self.rpo_alpha > 0:  # RPO algorithm
                # sample again to add stochasticity, for the policy update
                z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha)
                action_mean = action_mean + z
                probs = Normal(action_mean, action_std)
        # print(probs.log_prob(action))
        # print(probs.log_prob(action).shape)
        # quit()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


    def _get_action_and_value_discrete(self, x, action=None):
        # x = torch.atleast_2d(x)   # adds a batch dimension if there's only one

        logits = self.actor_mean(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()    #.squeeze()  # not using vector env

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


    def build_network(self, env, num_hidden, layer_norm, layer_norm_no_params, add_diag_layer,
                      activation, weight_init, init_gain, parseval_reg, input_scale, learnable_input_scale,
                      lie_group, oft, poet, oet_num_blocks, poet_exact_cayley, project_weight_dim, discrete_action_space):
        ''' '''
        if activation == 'tanh':
            activation_fn = nn.Tanh()
        elif activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'gelu':
            activation_fn = nn.GELU()
        elif activation == 'mish':
            activation_fn = nn.Mish()
        elif activation == 'leakyrelu':
            activation_fn = nn.LeakyReLU()
        elif activation == 'selu':
            activation_fn = nn.SELU()
        elif activation == 'silu':
            activation_fn = nn.SiLU()
        elif activation == 'groupsort':
            activation_fn = GroupSort(int(num_hidden / 2))
        elif activation == 'crelu':
            activation_fn = ConcatReLU()
        elif activation == 'identity' or activation is None:
            activation_fn = IdentityActivation()
        else:
            raise AssertionError('Invalid activation', activation)

        if weight_init == 'orthogonal':
            layer_init = orthogonal_layer_init
        elif weight_init == 'xavier':
            layer_init = xavier_layer_init
        elif weight_init == 'standard':
            layer_init = standard_layer_init
        elif weight_init == 'lecun':
            layer_init = lecun_layer_init
        elif weight_init == 'identity':
            layer_init = identity_layer_init
        elif weight_init == 'ones':
            layer_init = ones_layer_init
        else:
            raise AssertionError('weight_init is invalid:', weight_init)

        layer_name = 'linear_orthog' if parseval_reg > 0 else 'linear'

        def hidden_linear(in_features, out_features):
            if lie_group:
                return StiefelLinear(in_features, out_features, bias=True)
            if poet:
                return PoetOfficialLinear(in_features, out_features, bias=True, exact_cayley=poet_exact_cayley,
                                          weight_init=weight_init)
            if oft:
                return OETLinear(in_features, out_features, two_sided=False, bias=True, num_blocks=oet_num_blocks)
            return layer_init(nn.Linear(in_features, out_features), gain=init_gain)

        if activation == 'crelu':
            assert num_hidden % 2 == 0
            num_hidden_out = int(num_hidden / 2)
        else:
            num_hidden_out = num_hidden

        actor_output_dim = env.action_space.n if discrete_action_space else np.prod(env.action_space.shape)

        if layer_norm:
            critic = nn.Sequential(OrderedDict([
                ('input_scale', ScaleLayer(input_scale, learnable_input_scale)),
                (f'{layer_name}_1', hidden_linear(np.array(env.observation_space.shape).prod(), num_hidden_out)),
                ('layernorm_1', nn.LayerNorm(num_hidden_out, elementwise_affine=not layer_norm_no_params)),
                (f'{activation}_1', activation_fn),
                (f'{layer_name}_2', hidden_linear(num_hidden, num_hidden_out)),
                ('layernorm_2', nn.LayerNorm(num_hidden_out, elementwise_affine=not layer_norm_no_params)),
                (f'{activation}_2',activation_fn),
                ('linear_output', layer_init(nn.Linear(num_hidden, 1), gain=1.0)),
            ]))
            actor_mean = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(input_scale, learnable_input_scale)),
                (f'{layer_name}_1', hidden_linear(np.array(env.observation_space.shape).prod(), num_hidden_out)),
                ('layernorm_1', nn.LayerNorm(num_hidden_out, elementwise_affine=not layer_norm_no_params)),
                (f'{activation}_1', activation_fn),
                (f'{layer_name}_2', hidden_linear(num_hidden, num_hidden_out)),
                ('layernorm_2', nn.LayerNorm(num_hidden_out, elementwise_affine=not layer_norm_no_params)),
                (f'{activation}_2',activation_fn),
                ('linear_output', layer_init(nn.Linear(num_hidden, actor_output_dim), gain=0.01)),
            ]))
        elif add_diag_layer:
            critic = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(input_scale, learnable_input_scale)),
                (f'{layer_name}_1', hidden_linear(np.array(env.observation_space.shape).prod(), num_hidden_out)),
                ('diag_1', DiagLinear(num_hidden_out)),
                (f'{activation}_1', activation_fn),
                (f'{layer_name}_2', hidden_linear(num_hidden, num_hidden_out)),
                ('diag_2', DiagLinear(num_hidden_out)),
                (f'{activation}_2',activation_fn),
                ('linear_output', layer_init(nn.Linear(num_hidden, 1), gain=1.0)),
            ]))
            actor_mean = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(input_scale)),
                (f'{layer_name}_1', hidden_linear(np.array(env.observation_space.shape).prod(), num_hidden_out)),
                ('diag_1', DiagLinear(num_hidden_out)),
                (f'{activation}_1', activation_fn),
                (f'{layer_name}_2', hidden_linear(num_hidden, num_hidden_out)),
                ('diag_2', DiagLinear(num_hidden_out)),
                (f'{activation}_2',activation_fn),
                ('linear_output', layer_init(nn.Linear(num_hidden, actor_output_dim), gain=0.01)),
            ]))
        else:
            critic = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(input_scale, learnable_input_scale)),
                (f'{layer_name}_1', hidden_linear(np.array(env.observation_space.shape).prod(), num_hidden_out)),
                (f'{activation}_1', activation_fn),
                (f'{layer_name}_2', hidden_linear(num_hidden, num_hidden_out)),
                (f'{activation}_2', activation_fn),
                ('linear_output', layer_init(nn.Linear(num_hidden, 1), gain=1.0)),
            ]))
            actor_mean = nn.Sequential(OrderedDict( [
                ('input_scale', ScaleLayer(input_scale, learnable_input_scale)),
                (f'{layer_name}_1', hidden_linear(np.array(env.observation_space.shape).prod(), num_hidden_out)),
                (f'{activation}_1', activation_fn),
                (f'{layer_name}_2', hidden_linear(num_hidden, num_hidden_out)),
                (f'{activation}_2', activation_fn),
                ('linear_output', layer_init(nn.Linear(num_hidden, actor_output_dim), gain=0.01)),
            ]))
        return actor_mean, critic
    

if __name__ == "__main__":

    from envs.gridworld_env import NineRoomsEnv
    env = NineRoomsEnv()
    layer_norm=True
    nets = AgentNetworks(env, layer_norm=layer_norm)

    for name, param in nets.critic.named_parameters():
        if 'weight' in name and param.requires_grad:
            if layer_norm and name[0] in ['1', '4']:
                continue  # have to skip layer norm's parameters

            last_layer = '6'
            # print(name, param.shape)
            if name[0] == last_layer:  # last layer
                scale = 1
            else:
                scale = 2  # sqrt(2)**2
            # weight matrices right multiply by their inputs
            # print(torch.norm(torch.matmul(param, param.t()) - scale* torch.eye(param.shape[0]), p='fro')**2)
                # print(torch.matmul(param, param.t()))

            # extract weights this way
            # only regularize the weights but not the biases
            # (is that correct?)
    #
    # print('second)')
    # for key, item in nets.actor.state_dict().items():
    #     print(key, item.shape)