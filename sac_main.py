import pybullet_envs
import gym
import numpy as np
from sac_agent import Agent
from utilities import plot_learning_curve

if __name__ == '__main__':
    env_name = 'InvertedPendulumBulletEnv-v0'
    env = gym.make(env_name)
    
    agent = Agent(eta2=0.0003, 
                  eta1=0.0003, 
                  temperature=2, 
                  env_name=env_name, 
                  input_shape=env.observation_space.shape, 
                  tau=0.005,
                  env=env, 
                  batch_size=256, 
                  layer1_size=256, 
                  layer2_size=156,
                  action_space_dim=env.action_space.shape[0])
    
    n_games = 250
    filename = env_name + '_' + str(n_games) + 'games_temperature' + str(agent.temperature) + '.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_net_weights = True

    if load_net_weights:
        agent.load_networks()
        env.render(mode='human')
        
    steps = 0
    
    for i in range(n_games):
        
        score = 0
        done = False
        observation = env.reset()
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info =  env.step(action)
            steps += 1
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_net_weights:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            if not load_net_weights:
                agent.save_networks()
        print('episode ', i, 'score %.1f' % score, 'trailing 100 games average %.1f' % avg_score,
              'step %d' % steps, env_name, 'temperature', agent.temperature)
    if not load_net_weights:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
