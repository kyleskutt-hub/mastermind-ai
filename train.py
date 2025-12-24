import argparse
from mastermind import MastermindEnv, MastermindAgent

def train(episodes: int = 1000, code_length: int = 4, num_colors: int = 6):
    env = MastermindEnv(code_length=code_length, num_colors=num_colors)
    agent = MastermindAgent(code_length=code_length, num_colors=num_colors)
    
    for episode in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            agent.rewards.append(reward)
            total_reward += reward
        
        loss = agent.update()
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Reward: {total_reward:.2f}, Loss: {loss:.4f}")

def benchmark():
    print("Running benchmark...")
    env = MastermindEnv()
    agent = MastermindAgent()
    wins = 0
    for _ in range(100):
        obs = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
        if info["black"] == env.code_length:
            wins += 1
    print(f"Win rate (untrained): {wins}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--code-length", type=int, default=4)
    parser.add_argument("--num-colors", type=int, default=6)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark()
    else:
        train(args.episodes, args.code_length, args.num_colors)
