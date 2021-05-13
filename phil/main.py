import argparse 
# import os
import numpy as np
import agents as Agents
# from gym import wrappers
from util import plot_learning_curve, make_env

#-----------------------------------------------------------------------------------#

def main(args):
    
    #os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    env = make_env(args.env)
    
    best_score = -np.inf
    
    agent_ = getattr(Agents, args.algo)
    agent = agent_(gamma=args.gamma, 
                             epsilon=args.eps, 
                             lr=args.lr, 
                             input_dims=env.observation_space.shape,
                             n_actions=env.action_space.n,
                             mem_size=args.max_mem, 
                             eps_min=args.eps_min, 
                             batch_size=args.bs, 
                             replace=args.replace, 
                             eps_dec=args.eps_dec, 
                             chkpt_dir=args.path, 
                             algo=args.algo, 
                             env_name=args.env)
    
    if args.load_checkpoint:
        agent.load_models()
        
    # env = wrappers.Monitor(env, 
    #                        'tmp/video', 
    #                        video_callable=lambda episode_id: True,
    #                        force=True)
        
    fname = args.algo + '_' + args.env + '_lr' + str(args.lr) + '_' + str(args.n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    
    n_steps = 0
    scores, eps_history, steps_array = [], [], []
    
    for i in range(args.n_games):
        done = False
        score = 0
        observation = env.reset()
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            #env.render()
            if not args.load_checkpoint:
                agent.store_transition(observation, action, reward, observation, int(done))
                agent.learn()
                
            observation = observation_
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)
        
        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score: ', score, 'average score %.1f best score %.1f epsilon %.2f' % (avg_score, best_score, agent.epsilon), 'steps ', n_steps)

        if avg_score > best_score:
            if not args.load_checkpoint:
                agent.save_models()
            best_score = avg_score
            
        eps_history.append(agent.epsilon)
        
    plot_learning_curve(steps_array, scores, eps_history, figure_file)

#-----------------------------------------------------------------------------------#

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Deep Q-Learning')
    
    parser.add_argument('-n_games', type=int, default=1, help='Number of games to play')
    parser.add_argument('-lr', type=float, default=0.0001, help='Learning rate for optimizer')
    parser.add_argument('-eps_min', type=float, default=0.1, help='Minimum value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-gamma', type=float, default=0.99, help='Discount factor for update equation.')
    parser.add_argument('-eps_dec', type=float, default=1e-5, help='Linear factor for decreasing epsilon')
    parser.add_argument('-eps', type=float, default=1.0, help='Starting value for epsilon in epsilon-greedy action selection')
    parser.add_argument('-max_mem', type=int, default=6000, help='Maximum size for memory replay buffer')
    parser.add_argument('-skip', type=int, default=4, help='Number of frames to stack for environment')
    parser.add_argument('-bs', type=int, default=32, help='Batch size for replay memory sampling')
    parser.add_argument('-replace', type=int, default=1000, help='interval for replacing target network')
    parser.add_argument('-env', type=str, default='PongNoFrameskip-v4', help='Atari environment.\n \
                                                                              PongNoFrameskip-v4\n \
                                                                              BreakoutNoFrameskip-v4\n \
                                                                              SpaceInvadersNoFrameskip-v4\n \
                                                                              EnduroNoFrameskip-v4\n \
                                                                              AtlantisNoFrameskip-v4')
    #parser.add_argument('-gpu', type=str, default='0', help='GPU: 0 or 1')
    parser.add_argument('-load_checkpoint', type=bool, default=False, help='load model chakpoint')
    parser.add_argument('-path', type=str, default='tmp/', help='path for model saving/loading')
    parser.add_argument('-algo', type=str, default='DQNAgent', help='DQNAgent/DDQNAgent/DuelingDQNAgent/DuelingDDQNAgent')
    
    args = parser.parse_args()
                                
    
    main(args)