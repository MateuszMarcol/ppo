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

from ppo_base import Critic, ReplayBuffer, Actor, compute_average_return


def _make_batched_step(model, frame_skip):
    def step_fn(data, actions):
        def body(i, d):
            return mjx.step(model, d)
        def one(d, a):
            d = d.replace(ctrl=a)
            d = jax.lax.fori_loop(0, frame_skip, body, d)
            return d
        return jax.vmap(one)(data, actions)
    return jax.jit(step_fn)

class MJXVectorEnv:
    def __init__(self, model_path: str, n_envs: int, frame_skip: int = 4, seed: int = 0):
        # Load MuJoCo model in CPU (numpy) and convert to mjx (JAX)
        self.mj_model = mujoco.MjModel.from_xml_path(model_path)
        self.model = mjx.put_model(self.mj_model)  # JAX device arrays
        self.n_envs = n_envs
        self.frame_skip = frame_skip
        self.key = jax.random.PRNGKey(seed)

        # Create a single unbatched data, then replicate to batch as a pytree
        d0 = mjx.make_data(self.model)
        self.data = jax.tree.map(lambda x: jnp.repeat(x[None, ...], n_envs, axis=0), d0)

        # Control limits (nu x 2)
        self.ctrl_low = jnp.asarray(self.mj_model.actuator_ctrlrange[:, 0])
        self.ctrl_high = jnp.asarray(self.mj_model.actuator_ctrlrange[:, 1])

        # Sizes
        self.nq = self.mj_model.nq
        self.nv = self.mj_model.nv
        self.nu = self.mj_model.nu

        # Compile batched step once (closure captures model & frame_skip)
        self._step_jit = _make_batched_step(self.model, self.frame_skip)

    def _sample_init_state(self, key, batch):
        # Initialize around qpos0 with small noise; zero velocities
        qpos0 = jnp.asarray(self.mj_model.qpos0)  # [nq]
        key, kq, kv = jax.random.split(key, 3)
        qpos = qpos0 + 0.01 * jax.random.normal(kq, (batch, self.nq))
        qvel = 0.01 * jax.random.normal(kv, (batch, self.nv))
        return key, qpos, qvel

    def reset(self, key=None):
        if key is not None:
            self.key = key
        self.key, qpos, qvel = self._sample_init_state(self.key, self.n_envs)
        self.data = self.data.replace(qpos=qpos, qvel=qvel)
        # Warm-up compile with a dummy step to avoid first-step stall
        dummy_actions = jnp.zeros((self.n_envs, self.nu), dtype=self.data.ctrl.dtype)
        self.data = self._step_jit(self.data, dummy_actions)
        obs = self._obs(self.data)
        return obs, {}

    def reset_done(self, done_mask: jnp.ndarray):
        # done_mask: [n_envs] booleans
        num_done = int(jnp.sum(done_mask))
        if num_done == 0:
            return
        idx = jnp.where(done_mask)[0]
        self.key, qpos, qvel = self._sample_init_state(self.key, num_done)
        # Scatter qpos/qvel into the batch where done=True
        self.data = self.data.replace(
            qpos=self.data.qpos.at[idx].set(qpos),
            qvel=self.data.qvel.at[idx].set(qvel),
        )

    def step(self, actions: jnp.ndarray):
        # actions: [n_envs, nu]
        actions = jnp.clip(actions, self.ctrl_low, self.ctrl_high)
        self.data = self._step_jit(self.data, actions)
        obs = self._obs(self.data)
        reward, done = self._reward_done(self.data, actions)
        info = {}
        return obs, reward, done, info
    
    def _obs(self, data):
        # Simple observation: concat qpos (optionally drop root x) and qvel
        # For Hopper, Gym usually uses [qpos[1:], qvel], possibly clipped.
        qpos = data.qpos
        qvel = data.qvel
        obs = jnp.concatenate([qpos[:, 1:], qvel], axis=-1)
        return obs

    def _reward_done(self, data, actions):
        # Placeholder reward/done for Hopper-like task:
        # forward_vel ~ qvel[:, 0], alive bonus if height (qpos index 1) within range,
        # small control penalty.
        qpos = data.qpos
        qvel = data.qvel
        forward_vel = qvel[:, 0]
        height = qpos[:, 1]
        alive = (height > 0.7) & (height < 2.0)
        ctrl_cost = 0.001 * jnp.sum(actions**2, axis=-1)
        reward = forward_vel + 1.0 * alive.astype(jnp.float32) - ctrl_cost
        done = ~alive
        # NaN/infs safety
        bad = jnp.isnan(qpos).any(axis=-1) | jnp.isnan(qvel).any(axis=-1)
        done = done | bad
        return reward, done

# -------------------- PPO losses (pure, local) --------------------

def make_loss_fns(actor: Actor, critic: Critic, epsilon: float, entropy_coeff: float, value_coef: float):
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

    def loss_fn_critic(params_critic, batch_states, batch_returns):
        values = critic.apply(params_critic, batch_states).squeeze(-1)
        return 0.5 * jnp.mean((batch_returns - values) ** 2)
    
    def loss_fn_entropy(params_actor, batch_states):
        entropy = actor.get_entropy(params_actor, batch_states)
        return entropy

    def loss_fn(params_actor, params_critic, batch_states, batch_actions, batch_advantages, batch_old_log_probs, batch_returns):
        actor_loss, approx_kl = loss_fn_actor(params_actor, batch_states, batch_actions, batch_advantages, batch_old_log_probs)
        critic_loss = loss_fn_critic(params_critic, batch_states, batch_returns)
        entropy_loss = loss_fn_entropy(params_actor, batch_states)
        total = actor_loss + value_coef * critic_loss - entropy_coeff * entropy_loss
        return total, (actor_loss, critic_loss, approx_kl)
    
    def loss_fn_distributional_critic(params_critic, batch_states):
        values = critic.apply(params_critic, batch_states).squeeze(-1)
        return 0.5 * jnp.mean((batch_returns - values) ** 2)

    return loss_fn, loss_fn_actor, loss_fn_critic


def make_update_step(optimizer, loss_fn):
    @jax.jit
    def update_step(params_actor, params_critic, opt_state_actor, opt_state_critic,
                    batch_states, batch_actions, batch_advantages,
                    batch_old_log_probs, batch_returns):
        (loss, aux), grads = jax.value_and_grad(loss_fn, argnums=(0,1), has_aux=True)(
            params_actor, params_critic, batch_states, batch_actions,
            batch_advantages, batch_old_log_probs, batch_returns
        )
        actor_loss, critic_loss, approx_kl = aux
        grads_actor, grads_critic = grads

        updates_actor, opt_state_actor = optimizer.update(grads_actor, opt_state_actor, params_actor)
        params_actor = optax.apply_updates(params_actor, updates_actor)

        updates_critic, opt_state_critic = optimizer.update(grads_critic, opt_state_critic, params_critic)
        params_critic = optax.apply_updates(params_critic, updates_critic)

        return params_actor, params_critic, opt_state_actor, opt_state_critic, (loss, actor_loss, critic_loss, approx_kl)
    return update_step

# -------------------- Training script --------------------

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
    entropy_coeff = 0.1
    value_coef = 0.5
    learning_rate = 1e-4
    target_kl = 0.05

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
    critic = Critic(state_dim=state_dim)
    buffer = ReplayBuffer(discount=discount, gae_lambda=gae_lambda,
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
            buffer.add(state_jax, action, reward, done)

            # Reset finished envs to keep rollout going
            env.reset_done(done)
            obs = next_obs

        # Compute targets
        buffer.compute_advantages(params_critic, critic)
        buffer.compute_returns()
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
                batch_returns = buffer.flat_returns[batch_idx]

                params_actor, params_critic, opt_state_actor, opt_state_critic, (loss, actor_loss, critic_loss, approx_kl) = update_step(
                    params_actor, params_critic, opt_state_actor, opt_state_critic,
                    batch_states, batch_actions, batch_advantages,
                    batch_old_log_probs, batch_returns
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

