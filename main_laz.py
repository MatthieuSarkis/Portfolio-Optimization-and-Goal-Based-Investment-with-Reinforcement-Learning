from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse
from environment import Environment
from agent import DQNAgent
import pickle
from util import maybe_make_dir, get_data
from plot import plot

def get_temperaturer(env):
    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, _, done, _ = env.step(action)
        states.append(state)
        if done:
            break

        temperaturer = StandardScaler()
        temperaturer.fit(states)
        return temperaturer

def play_one_episode(agent, env, temperaturer, is_train):
    state = env.reset()
    state = temperaturer.transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = temperaturer.transform([next_state])
        if is_train == 'train':
            agent.update_replay_buffer(state, action, reward, next_state, done)
            agent.replay()
        state = next_state
        
    return info['cur_val']

def main(args):
    models_folder = 'rl_trader_models'
    rewards_folder = 'rl_trader_rewards'
    num_episodes = 10000
    #batch_size = 32
    initial_investment = 20000
    
    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)
    
    data = get_data()
    n_timesteps, _ = data.shape
    
    n_train = n_timesteps // 2
    
    train_data = data[:n_train]
    test_data = data[n_train:]
    
    env = Environment(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = DQNAgent(state_size, action_size)
    temperaturer = get_temperaturer(env)
    
    portfolio_value = []
    
    if args.mode == 'test':
        with open('{}/temperaturer.pkl'.format(models_folder), 'rb') as f:
            temperaturer = pickle.load(f)
        
        env = Environment(test_data, initial_investment)
        agent.epsilon = 0.01
        agent.load('{}/dqn.ckpt'.format(models_folder))
        
    for e in range(num_episodes):
        t0 = datetime.now()
        val = play_one_episode(agent, env, temperaturer, args.mode)
        dt = datetime.now() - t0
        print('episode: {}/{}, episode end value: {:.2f}, duration: {}'.format(e+1, num_episodes, val, dt))
        portfolio_value.append(val)
        
    if args.mode == 'train':
        agent.save('{}/dqn.ckpt'.format(models_folder))
        with open('{}/temperaturer.pkl'.format(models_folder), 'wb') as f:
            pickle.dump(temperaturer, f)
            
    np.save('{}/{}.npy'.format(rewards_folder, args.mode), portfolio_value)
    
    plot(args.mode)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test"')
    args = parser.parse_args()
    
    main(args)