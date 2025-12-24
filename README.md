# 🎯 Mastermind AI

A reinforcement learning environment and PyTorch agent for the classic code-breaking game **Mastermind**.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 🎮 What is Mastermind?

Mastermind is a code-breaking game where one player creates a secret code of colored pegs, and the other player tries to guess it. After each guess, feedback is given:

- **⚫ Black peg**: Correct color in correct position
- **⚪ White peg**: Correct color in wrong position

The goal is to crack the code in as few guesses as possible.

```
SECRET CODE:  🔴 🔵 🟢 🟡  (hidden)

GUESS 1:      🔴 🔴 🟢 🔵  →  ⚫⚫⚪ (2 exact, 1 color match)
GUESS 2:      🔴 🔵 🟢 🟡  →  ⚫⚫⚫⚫ WIN!
```

## 🚀 Features

- **Clean RL Environment**: OpenAI Gym-style interface for training agents
- **PyTorch Neural Network**: Actor-Critic architecture with policy gradient
- **Multiple Agents**: 
  - Neural network (trainable)
  - Random baseline
  - Minimax optimal (benchmark)
- **Full Training Pipeline**: Train, evaluate, and benchmark
- **Interactive Mode**: Watch the AI play

## 📦 Installation

```bash
git clone https://github.com/kyleskutt-hub/mastermind-ai.git
cd mastermind-ai
pip install -e .
```

Or with dev dependencies:

```bash
pip install -e ".[dev]"
```

## 🏃 Quick Start

### Train an Agent

```bash
python train.py --episodes 10000
```

### Watch AI Play

```bash
python train.py --play
```

### Benchmark Agents

```bash
python train.py --benchmark
```

## 📊 Results

| Agent | Win Rate | Avg Guesses |
|-------|----------|-------------|
| Random | ~0% | 10+ |
| Neural Net (10k episodes) | ~65% | 6.2 |
| Neural Net (50k episodes) | ~85% | 5.1 |
| Minimax (Optimal) | 100% | 4.3 |

*Human average: ~5-6 guesses*

## 🧠 How It Works

### Environment

The environment tracks:
- Secret code (hidden from agent)
- Guess history
- Feedback history

**State representation**: Normalized matrix of previous guesses and their feedback.

**Action space**: All possible 4-color combinations (6^4 = 1296 actions).

**Rewards**:
- +1.0 per black peg
- +0.5 per white peg
- +10.0 for winning (scaled by efficiency)
- -5.0 for running out of guesses

### Agent Architecture

```
Input (state) → [256] → [256] → [128] → Policy Head (1296 actions)
                                    └─→ Value Head (1 value)
```

Uses Actor-Critic with advantage estimation for stable training.

## 🔧 API Usage

```python
from mastermind import MastermindEnv, MastermindAgent

# Create environment
env = MastermindEnv()

# Create agent
agent = MastermindAgent(
    state_size=env.state_size,
    action_size=env.action_space_size
)

# Play one game
state = env.reset()
done = False

while not done:
    action = agent.select_action(state)
    state, reward, done, info = env.step(action)
    print(env.render())

print(f"Won: {env.won}, Guesses: {len(env.guess_history)}")
```

## 📁 Project Structure

```
mastermind-ai/
├── mastermind/
│   ├── __init__.py      # Package exports
│   ├── environment.py   # Game environment
│   └── agent.py         # Neural network agent
├── train.py             # Training script
├── checkpoints/         # Saved models
├── pyproject.toml       # Package config
├── requirements.txt     # Dependencies
└── README.md
```

## 🛠️ Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black mastermind/ train.py
isort mastermind/ train.py

# Type checking
mypy mastermind/
```

## 📈 Training Tips

1. **Start small**: 10k episodes gets you to ~65% win rate
2. **Learning rate**: 3e-4 works well, lower for fine-tuning
3. **Hidden layers**: [256, 256, 128] balances capacity and speed
4. **Patience**: Optimal play requires ~50k+ episodes

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📄 License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Classic Mastermind game by Mordecai Meirowitz (1970)
- Knuth's minimax algorithm (1977)
- PyTorch team for the excellent framework

---

**Author**: Kyle Skutt  
**GitHub**: [@kyleskutt-hub](https://github.com/kyleskutt-hub)
