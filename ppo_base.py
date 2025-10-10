import jax
import flax.linen as nn
import jax.numpy as jnp
import gymnasium as gym
import optax
import numpy as np
from tqdm import tqdm
import mujoco
from mujoco import mjx
import distrax

class Critic(nn.Module):
    state_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64, kernel_init = nn.initializers.orthogonal())(x)
        x = nn.tanh(x)
        x = nn.Dense(64, kernel_init = nn.initializers.orthogonal())(x)
        x = nn.tanh(x)
        x = nn.Dense(1, kernel_init = nn.initializers.orthogonal())(x)
        return x
    
    def get_value(self, params, state):
        value = self.apply(params, state)
        return value
    


class Actor(nn.Module):
    state_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64, kernel_init = nn.initializers.orthogonal())(x)
        x = nn.tanh(x)
        x = nn.Dense(64, kernel_init = nn.initializers.orthogonal())(x)
        x = nn.tanh(x)
        mu = nn.Dense(self.action_dim, kernel_init = nn.initializers.orthogonal())(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        std = jnp.exp(log_std)
        return mu, std 

    def get_action(self, params, state, key):
        mu, std = self.apply(params, state)
        noise = jax.random.normal(key, mu.shape)
        action = mu + std * noise
        return action

    def get_entropy(self, params, state):
        mu, std = self.apply(params, state)
        pi = distrax.MultivariateNormalDiag(mu, std)
        return pi.entropy().mean()
    


class ReplayBuffer():
    def __init__(
        self,
        discount: float,
        gae_lambda: float,
        num_steps_per_environment: int,
        n_envs: int,
        state_dim: int,
        action_dim: int
    ):
        self.num_steps = num_steps_per_environment
        self.num_envs = n_envs
        self.buffer_size = self.num_steps * self.num_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.ptr = 0


    def reset(self):
        # Store rollout data in NumPy to avoid per-step JAX scatters
        self.states = np.zeros((self.num_steps, self.num_envs, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.num_steps, self.num_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        self.dones = np.zeros((self.num_steps, self.num_envs), dtype=np.float32)
        # Computed tensors will be JAX arrays
        self.advantages = None
        self.log_probs = None
        self.ptr = 0


    def add(self, state, action, reward, done):
        idx = self.ptr % self.num_steps  # index over time steps only
        # Convert device arrays to host once per step
        self.states[idx] = np.asarray(state)
        self.actions[idx] = np.asarray(action)
        self.rewards[idx] = np.asarray(reward)
        self.dones[idx] = np.asarray(done, dtype=np.float32)
        self.ptr += 1

    def compute_advantages(self, params_critic, critic):
        # Convert to JAX once
        states = jnp.asarray(self.states)
        rewards = jnp.asarray(self.rewards)
        dones = jnp.asarray(self.dones)
        all_values = jnp.squeeze(critic.apply(params_critic, states), axis=-1)
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

    def compute_returns(self):
        self.returns = self.advantages + self.values
        self.flat_returns = self.returns.reshape(-1)

    def compute_log_probs(self, params_actor, actor, key):
        states = jnp.asarray(self.states)
        actions = jnp.asarray(self.actions)
        mu, std = actor.apply(params_actor, states)
        log_std = jnp.log(std)
        log_probs = -0.5 * (((actions - mu) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))
        self.log_probs = log_probs.sum(axis=-1)[:-1]
        self.flat_log_probs = self.log_probs.reshape(-1)


    def flatten(self):
        # Create JAX arrays for training
        self.flat_states = jnp.asarray(self.states[:-1]).reshape(-1, self.state_dim)
        self.flat_actions = jnp.asarray(self.actions[:-1]).reshape(-1, self.action_dim)
        self.flat_rewards = jnp.asarray(self.rewards[:-1]).reshape(-1)
        self.flat_dones = jnp.asarray(self.dones[:-1]).reshape(-1)
        self.flat_advantages = self.advantages.reshape(-1)

    def normalize_advantages(self):
        mean = self.flat_advantages.mean()
        std = self.flat_advantages.std()
        self.flat_advantages = (self.flat_advantages - mean) / (std + 1e-8)

    def save_old_flat_log_probs(self):
        self.flat_old_log_probs = self.flat_log_probs

def compute_average_return(flat_rewards, flat_dones):
    # flat_rewards: shape [T*N]
    # flat_dones: shape [T*N], True where episode ended
    episode_returns = []
    current_return = 0.0
    for r, d in zip(flat_rewards, flat_dones):
        current_return += r
        if d:
            episode_returns.append(current_return)
            current_return = 0.0
    # If last episode didn't terminate, you can optionally include it:
    if current_return != 0.0:
        episode_returns.append(current_return)
    return np.mean(episode_returns) if episode_returns else 0.0



def loss_fn_actor(params_actor, batch_states, batch_actions, batch_advantages, batch_old_log_probs):
    mu, std = actor.apply(params_actor, batch_states)
    log_probs = -0.5 * (((batch_actions - mu) / std) ** 2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))
    log_probs = log_probs.sum(axis=-1)
    log_ratio = log_probs - batch_old_log_probs
    ratio = jnp.exp(log_ratio)
    pg_loss1 = -batch_advantages * ratio
    pg_loss2 = -batch_advantages * jnp.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
    pg_loss = jnp.maximum(pg_loss1, pg_loss2)
    entropy = (0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(std)).sum(axis=-1).mean()
    actor_loss = pg_loss.mean() - entropy_coeff * entropy
    approx_kl = -jnp.mean(log_ratio)
    return actor_loss, approx_kl

def loss_fn_critic(params_critic, batch_states, batch_returns):
    values = critic.apply(params_critic, batch_states).squeeze(-1)
    return 0.5 * jnp.mean((batch_returns - values) ** 2)

def loss_fn(params_actor, params_critic, batch_states, batch_actions, batch_advantages, batch_old_log_probs, batch_returns):
    actor_loss, approx_kl = loss_fn_actor(params_actor, batch_states, batch_actions, batch_advantages, batch_old_log_probs)
    critic_loss = loss_fn_critic(params_critic, batch_states, batch_returns)
    total = actor_loss + value_coef * critic_loss
    return total, (actor_loss, critic_loss, approx_kl)

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




if __name__ == "__main__":
    print("JAX version:", jax.__version__)
    print("All dependencies installed correctly!")

    device = jax.devices()[0]
    print(f"Using device: {device}")

    env_name = "Hopper-v5"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]


    # initialize hyperparameters
    num_iterations = 1000
    num_steps_per_environment = 1024 # timesteps per environment per iteration
    n_envs = 16
    num_epochs = 10
    discount = 0.99
    gae_lambda = 0.95
    batch_size = 64
    epsilon = 0.2
    entropy_coeff = 0.01
    value_coef = 0.5
    learning_rate = 3e-4


    # prepare environement setup
    envs = gym.vector.AsyncVectorEnv([lambda: gym.make(env_name) for _ in range(n_envs)])

    assert isinstance(envs.single_action_space, gym.spaces.Box), "Only continuous action space is supported"

    actor = Actor(state_dim=state_dim, action_dim = action_dim)
    critic = Critic(state_dim=state_dim)
    buffer = ReplayBuffer(discount=discount, gae_lambda=gae_lambda, num_steps_per_environment=num_steps_per_environment, n_envs=n_envs, state_dim=state_dim, action_dim=action_dim)
    

    optimizer = optax.adam(learning_rate=learning_rate)

    # initialize the networks
    params_actor = actor.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)))
    params_critic = critic.init(jax.random.PRNGKey(1), jnp.zeros((1, state_dim)))

    # initialize the optimizers
    opt_state_actor = optimizer.init(params_actor)
    opt_state_critic = optimizer.init(params_critic) 

    global_step = 0
    key = jax.random.PRNGKey(global_step)
  

    # main training loop
    for iteration in tqdm(range(num_iterations)):

        state, info = envs.reset() # reset the environment
        print(f'Iteration number {iteration}')
        print(f'Collecting data from the environment...')
        
        # loop over steps
        buffer.reset()
        for step in tqdm(range(num_steps_per_environment)):

            global_step += n_envs
            key, act_key = jax.random.split(key)
            state_jax = jnp.array(state)
            action_probs = actor.apply(params_actor, state_jax)
            action = actor.get_action(params_actor, state_jax, act_key)
            action = jnp.clip(action, envs.single_action_space.low, envs.single_action_space.high)
            value = critic.apply(params_critic, state_jax)

            # take a step in all environments
            next_state, reward, done, truncated, info = envs.step(np.asarray(action).reshape(n_envs, action_dim))
            next_done = jnp.logical_or(done, truncated)
            next_state_jax = jnp.array(next_state)
            buffer.add(state_jax, action, reward, next_done)
            state = next_state


        buffer.compute_advantages(params_critic, critic)
        buffer.compute_returns()
        buffer.compute_log_probs(params_actor, actor, key)
        buffer.flatten() 
        buffer.normalize_advantages()
        buffer.save_old_flat_log_probs()
        num_samples = buffer.flat_states.shape[0]

        avg_return = compute_average_return(np.array(buffer.flat_rewards), np.array(buffer.flat_dones))
        print(f"Average episodic return in buffer: {avg_return:.2f}")

        
        print(f'Upgrading the critic and the actor...')

        for epoch in range(num_epochs):
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, num_samples)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_idx = perm[start:end]

                batch_states = buffer.flat_states[batch_idx]
                batch_actions = buffer.flat_actions[batch_idx]
                batch_advantages = buffer.flat_advantages[batch_idx]
                batch_log_probs = buffer.flat_log_probs[batch_idx]
                batch_old_log_probs = buffer.flat_old_log_probs[batch_idx]
                batch_returns = buffer.flat_returns[batch_idx]

                params_actor, params_critic, opt_state_actor, opt_state_critic, aux = update_step(
                    params_actor, params_critic, opt_state_actor, opt_state_critic,
                    batch_states, batch_actions, batch_advantages,
                    batch_old_log_probs, batch_returns
                )