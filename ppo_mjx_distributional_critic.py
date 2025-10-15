import jax
import flax.linen as nn
import jax.numpy as jnp
import gymnasium as gym
import optax
import numpy as np
from tqdm import tqdm
import mujoco
from mujoco import mjx
import os
from datetime import datetime

from ppo_base import ReplayBuffer, Actor, compute_average_return
from ppo_mjx import MJXVectorEnv, _make_batched_step


# --------------- Distributional critic ---------------------
class DistributionalCritic(nn.Module):
    state_dim: int
    n_atoms: int = 101
    v_min: float = -100.0
    v_max: float = 3500.0
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64, kernel_init=nn.initializers.orthogonal())(x)
        x = nn.relu(x)
        x = nn.Dense(64, kernel_init=nn.initializers.orthogonal())(x)
        x = nn.relu(x)
        logits = nn.Dense(self.n_atoms, kernel_init=nn.initializers.orthogonal())(x)
        return logits
    
    def get_distribution(self, params, state):
        logits = self.apply(params, state)
        probs = jax.nn.softmax(logits, axis=-1)
        atoms = jnp.linspace(self.v_min, self.v_max, self.n_atoms)
        return probs, atoms
    
    def get_value(self, params, state):
        """Expected value from the distribution - used for advantages"""
        probs, atoms = self.get_distribution(params, state)
        return jnp.sum(probs * atoms, axis=-1)
    
    def sample_value(self, params, state, key):
        probs, atoms = self.get_distribution(params, state)
        batch_size = state.shape[0]
        
        # For each sample in batch, sample one atom index according to probabilities
        keys = jax.random.split(key, batch_size)
        
        def sample_one(prob_dist, rng_key):
            # Sample index from categorical distribution
            idx = jax.random.choice(rng_key, self.n_atoms, p=prob_dist)
            return atoms[idx]
        
        # Vectorized sampling for entire batch
        sampled_values = jax.vmap(sample_one)(probs, keys)
        
        return sampled_values
    
    def projection(
        self,
        params,
        next_states,
        rewards,
        dones,
        discount,
    ):
        """
        Project the Bellman update onto the categorical distribution.
        
        For each transition (s, r, s'):
        1. Get distribution Z(s') 
        2. Compute target atoms: Tz = r + γ * (1 - done) * z
        3. Project Tz back onto the fixed support
        """
        delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
        batch_size = rewards.shape[0]
        
        # Get next state distribution P(Z(s'))
        next_logits = self.apply(params, next_states)
        next_probs = jax.nn.softmax(next_logits, axis=-1)  # [batch, num_atoms]
        
        # Support atoms
        atoms = jnp.linspace(self.v_min, self.v_max, self.n_atoms)  # [num_atoms]
        
        # Compute target support: Tz = r + γ * (1 - done) * z
        bootstrap = 1.0 - dones  # [batch]
        target_z = rewards[:, None] + bootstrap[:, None] * discount * atoms[None, :]  # [batch, num_atoms]
        
        # Clip to support bounds
        target_z = jnp.clip(target_z, self.v_min, self.v_max)
        
        # Project onto categorical support
        # For each target atom, distribute its probability to neighboring atoms
        b = (target_z - self.v_min) / delta_z  # [batch, num_atoms]
        l = jnp.floor(b).astype(jnp.int32)
        u = jnp.ceil(b).astype(jnp.int32)
        
        # Handle edge cases
        is_int = (l == u)
        l_mask = is_int & (l > 0)
        u_mask = is_int & (l == 0)
        
        l = jnp.where(l_mask, l - 1, l)
        u = jnp.where(u_mask, u + 1, u)
        
        # Clamp indices
        l = jnp.clip(l, 0, self.n_atoms - 1)
        u = jnp.clip(u, 0, self.n_atoms - 1)
        
        # Distribute probability mass using linear interpolation
        u_weight = b - l.astype(jnp.float32)
        l_weight = 1.0 - u_weight
        
        # Initialize projected distribution
        proj_dist = jnp.zeros((batch_size, self.n_atoms), dtype=jnp.float32)
        
        # Accumulate probability mass
        batch_indices = jnp.arange(batch_size)[:, None]
        proj_dist = proj_dist.at[batch_indices, l].add(next_probs * l_weight)
        proj_dist = proj_dist.at[batch_indices, u].add(next_probs * u_weight)
        
        return proj_dist
    

class ReplayBufferDistributional(ReplayBuffer):
    def __init__(self, discount, gae_lambda, num_steps_per_environment, n_envs, state_dim, action_dim):
        super().__init__(discount, gae_lambda, num_steps_per_environment, n_envs, state_dim, action_dim)
        # Add storage for next_states
        self.next_states = []
        
    def add(self, state, action, reward, done, next_state):
        """Override to also store next_state"""
        super().add(state, action, reward, done)
        self.next_states.append(next_state)
    
    def flatten(self):
        """Override to also flatten next_states"""
        super().flatten()
        # Flatten next_states
        next_states_array = jnp.array(self.next_states)
        self.flat_next_states = next_states_array.reshape(-1, self.state_dim)
        
        # Also create single-step rewards and dones for distributional loss
        rewards_array = jnp.array(self.rewards)
        dones_array = jnp.array(self.dones)
        self.flat_rewards_single = rewards_array[:-1].flatten()  # Exclude last bootstrap step
        self.flat_dones_single = dones_array[:-1].flatten()
    
    def compute_advantages_distributional(self, params_critic, critic, sample=False):
        """Use expected value for advantage computation"""
        states = jnp.asarray(self.states)
        rewards = jnp.asarray(self.rewards)
        dones = jnp.asarray(self.dones)
        
        # Get expected values from distributional critic
        all_values = critic.get_value(params_critic, states)
        if sample: 
            all_values = critic.sample_value(params_critic, states)
        self.values = all_values[:-1]
        next_values = all_values[1:]
        
        not_done = 1.0 - dones[:-1]
        next_values = next_values * not_done
        self.delta = rewards[:-1] + self.discount * next_values - self.values

        def gae_carry(advantage, inputs):
            delta_t, not_done_t = inputs
            advantage = delta_t + self.discount * self.gae_lambda * not_done_t * advantage
            return advantage, advantage

        _, self.advantages = jax.lax.scan(
            gae_carry,
            init=jnp.zeros((self.num_envs,)),
            xs=(self.delta, not_done),
            reverse=True,
        )


def make_loss_fns(actor: Actor, critic: DistributionalCritic, epsilon: float, entropy_coeff: float, value_coef: float):
    def loss_fn_actor(params_actor, batch_states, batch_actions, batch_advantages, batch_old_log_probs):
        mu, std = actor.apply(params_actor, batch_states)
        log_probs = -0.5 * (((batch_actions - mu) / std) ** 2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))
        log_probs = log_probs.sum(axis=-1)
        log_ratio = log_probs - batch_old_log_probs
        ratio = jnp.exp(log_ratio)
        pg_loss1 = -batch_advantages * ratio
        pg_loss2 = -batch_advantages * jnp.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2)
        actor_loss = pg_loss.mean()
        approx_kl = -jnp.mean(log_ratio)
        return actor_loss, approx_kl

    def loss_fn_distributional_critic(params_critic, batch_states, batch_next_states, batch_rewards, batch_dones, discount):
        """
        Distributional critic loss using cross-entropy between:
        - Predicted distribution P(Z(s))
        - Projected target distribution based on r + γ * Z(s')
        """
        # Get predicted logits for current states
        current_logits = critic.apply(params_critic, batch_states)
        
        # Compute target distribution using projection
        target_dist = critic.projection(
            params_critic,
            batch_next_states,
            batch_rewards,
            batch_dones,
            discount
        )
        
        # Cross-entropy loss: -sum(target * log(predicted))
        log_probs = jax.nn.log_softmax(current_logits, axis=-1)
        loss = -jnp.sum(target_dist * log_probs, axis=-1).mean()
        
        return loss
    
    def loss_fn_entropy(params_actor, batch_states):
        entropy = actor.get_entropy(params_actor, batch_states)
        return entropy

    def loss_fn(params_actor, params_critic, batch_states, batch_actions, batch_advantages, 
                batch_old_log_probs, batch_next_states, batch_rewards, batch_dones, discount):
        actor_loss, approx_kl = loss_fn_actor(params_actor, batch_states, batch_actions, batch_advantages, batch_old_log_probs)
        critic_loss = loss_fn_distributional_critic(params_critic, batch_states, batch_next_states, batch_rewards, batch_dones, discount)
        entropy_loss = loss_fn_entropy(params_actor, batch_states)
        total = actor_loss + value_coef * critic_loss - entropy_coeff * entropy_loss
        return total, (actor_loss, critic_loss, approx_kl)

    return loss_fn, loss_fn_actor, loss_fn_distributional_critic


def make_update_step(optimizer, loss_fn):
    @jax.jit
    def update_step(params_actor, params_critic, opt_state_actor, opt_state_critic,
                    batch_states, batch_actions, batch_advantages,
                    batch_old_log_probs, batch_next_states, batch_rewards, batch_dones, discount):
        (loss, aux), grads = jax.value_and_grad(loss_fn, argnums=(0,1), has_aux=True)(
            params_actor, params_critic, batch_states, batch_actions,
            batch_advantages, batch_old_log_probs, batch_next_states, batch_rewards, batch_dones, discount
        )
        actor_loss, critic_loss, approx_kl = aux
        grads_actor, grads_critic = grads

        updates_actor, opt_state_actor = optimizer.update(grads_actor, opt_state_actor, params_actor)
        params_actor = optax.apply_updates(params_actor, updates_actor)

        updates_critic, opt_state_critic = optimizer.update(grads_critic, opt_state_critic, params_critic)
        params_critic = optax.apply_updates(params_critic, updates_critic)

        return params_actor, params_critic, opt_state_actor, opt_state_critic, (loss, actor_loss, critic_loss, approx_kl)
    return update_step


if __name__ == "__main__":
    print("JAX version:", jax.__version__)
    device = jax.devices()[0]
    print(f"Using device: {device}")

    # MuJoCo XML path for Hopper (adjust if needed)
    model_path = '/media/disk3/mateusz/ppo_clean/.venv/lib/python3.12/site-packages/gymnasium/envs/mujoco/assets/hopper.xml'

    # Hyperparameters
    num_iterations = 200
    num_steps_per_environment = 1024
    n_envs = 16
    num_epochs = 8
    discount = 0.99
    gae_lambda = 0.95
    batch_size = 64
    epsilon = 0.1
    entropy_coeff = 0.01
    value_coef = 0.5
    learning_rate = 1e-4
    target_kl = 0.05
    
    # Distributional critic parameters
    n_atoms = 101
    v_min = -100.0
    v_max = 3500.0

    # Potentiall improvements:
    # - add value clipping

    # Setup logging
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(results_dir, f'ppo_mjx_losses_{ts}.csv')
    with open(log_path, 'w') as f:
        f.write('iteration,epoch,global_step,loss,actor_loss,critic_loss,approx_kl,avg_return\n')

    # Env setup (infer dims)
    dummy_env = MJXVectorEnv(model_path, n_envs=1)
    key = jax.random.PRNGKey(0)
    dummy_obs, _ = dummy_env.reset(key)
    state_dim = int(dummy_obs.shape[-1])
    action_dim = int(dummy_env.nu)

    env = MJXVectorEnv(model_path, n_envs=n_envs)
    obs, _ = env.reset(key)

    # Models and buffer
    actor = Actor(state_dim=state_dim, action_dim=action_dim)
    critic = DistributionalCritic(state_dim=state_dim, n_atoms=n_atoms, v_min=v_min, v_max=v_max)
    buffer = ReplayBufferDistributional(discount=discount, gae_lambda=gae_lambda,
                          num_steps_per_environment=num_steps_per_environment,
                          n_envs=n_envs, state_dim=state_dim, action_dim=action_dim)

    optimizer = optax.adam(learning_rate=learning_rate)
    loss_fn, loss_fn_actor, loss_fn_critic = make_loss_fns(actor, critic, epsilon, entropy_coeff, value_coef)
    update_step = make_update_step(optimizer, loss_fn)

    # Init params and optimizer states
    params_actor = actor.init(jax.random.PRNGKey(1), jnp.zeros((1, state_dim)))
    params_critic = critic.init(jax.random.PRNGKey(2), jnp.zeros((1, state_dim)))
    opt_state_actor = optimizer.init(params_actor)
    opt_state_critic = optimizer.init(params_critic)

    global_step = 0

    for iteration in tqdm(range(num_iterations)):
        print(f"Iteration {iteration}")
        buffer.reset()
        obs, _ = env.reset(key)

        # Collect rollout
        for t in tqdm(range(num_steps_per_environment)):
            # removed per-timestep print to avoid slowdown
            global_step += n_envs
            key, act_key = jax.random.split(key)
            state_jax = jnp.array(obs)
            mu, std = actor.apply(params_actor, state_jax)
            noise = jax.random.normal(act_key, mu.shape)
            action = mu + std * noise
            action = jnp.clip(action, env.ctrl_low, env.ctrl_high)

            next_obs, reward, done, info = env.step(action)
            buffer.add(state_jax, action, reward, done, next_obs)

            # Reset finished envs to keep rollout going
            env.reset_done(done)
            obs = next_obs

        # Compute targets for distributional critic
        buffer.compute_advantages_distributional(params_critic, critic, sample=True)
        buffer.normalize_advantages()
        buffer.compute_log_probs(params_actor, actor, key)
        buffer.flatten()
        buffer.save_old_flat_log_probs()
        num_samples = buffer.flat_states.shape[0]

        avg_return = compute_average_return(np.array(buffer.flat_rewards), np.array(buffer.flat_dones))
        print(f"Average episodic return in buffer: {avg_return:.2f}")

        # PPO updates (use all samples per epoch)
        for epoch in range(num_epochs):
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, num_samples)

            # accumulate per-epoch averages
            loss_sum = 0.0
            actor_loss_sum = 0.0
            critic_loss_sum = 0.0
            kl_sum = 0.0
            batches = 0

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_idx = perm[start:end]

                batch_states = buffer.flat_states[batch_idx]
                batch_actions = buffer.flat_actions[batch_idx]
                batch_advantages = buffer.flat_advantages[batch_idx]
                batch_old_log_probs = buffer.flat_old_log_probs[batch_idx]
                batch_next_states = buffer.flat_next_states[batch_idx]
                batch_rewards_single = buffer.flat_rewards_single[batch_idx]
                batch_dones_single = buffer.flat_dones_single[batch_idx]

                # Call update with these
                params_actor, params_critic, opt_state_actor, opt_state_critic, (loss, actor_loss, critic_loss, approx_kl) = update_step(
                    params_actor, params_critic, opt_state_actor, opt_state_critic,
                    batch_states, batch_actions, batch_advantages, batch_old_log_probs,
                    batch_next_states, batch_rewards_single, batch_dones_single, discount
                )
                # accumulate
                loss_sum += float(loss)
                actor_loss_sum += float(actor_loss)
                critic_loss_sum += float(critic_loss)
                kl_sum += float(approx_kl)
                batches += 1

            # epoch averages
            loss_avg = loss_sum / max(1, batches)
            actor_loss_avg = actor_loss_sum / max(1, batches)
            critic_loss_avg = critic_loss_sum / max(1, batches)
            kl_avg = kl_sum / max(1, batches)

            # append to CSV with current global_step (timesteps collected so far)
            with open(log_path, 'a') as f:
                f.write(f"{iteration},{epoch},{global_step},{loss_avg},{actor_loss_avg},{critic_loss_avg},{kl_avg},{avg_return}\n")

            print(f"Epoch {epoch}: loss={loss_avg:.3f} actor={actor_loss_avg:.3f} critic={critic_loss_avg:.3f} kl={kl_avg:.6f}")

            # # ✓ Early stop if KL divergence too high
            # kl_avg = kl_sum / max(1, batches)
            # if abs(kl_avg) > target_kl:
            #     print(f"Early stopping at epoch {epoch}, KL={kl_avg:.6f} > {target_kl}")
            #     break



