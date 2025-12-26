import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import DeepNeuralNetwork
from src.utils import load_data

print("="*60)
print("REGULARIZATION SEARCH (LR=0.0075)")
print("="*60)

train_x, train_y, test_x, test_y, classes = load_data()

# Test different regularization combinations
configs = [
    # (dropout_rates, lambd, description)
    ([0, 0, 0, 0], 0.0, "Baseline (no reg)"),
    ([0, 0.2, 0.4, 0], 0.0, "Dropout only"),
    ([0, 0, 0, 0], 0.01, "L2 only (0.01)"),
    ([0, 0.2, 0.4, 0], 0.005, "Dropout + L2 (0.005)"),
    ([0, 0.3, 0.5, 0], 0.01, "Strong Dropout + L2"),
]

results = []

for dropout, lambd, desc in configs:
    print(f"\n[Testing: {desc}]")
    print(f"  Dropout: {dropout}, L2: {lambd}")
    
    model = DeepNeuralNetwork(
        [784, 128, 64, 24],
        learning_rate=0.0075,
        dropout_rates=dropout,
        lambd=lambd
    )

    model.train(train_x, train_y, num_iterations=10000, print_cost=False)
    
    train_acc = model.evaluate(train_x, train_y)
    test_acc = model.evaluate(test_x, test_y)
    gap = train_acc - test_acc
    
    results.append({
        'desc': desc,
        'train': train_acc,
        'test': test_acc,
        'gap': gap
    })
    
    print(f"  Train: {train_acc:.2%} | Test: {test_acc:.2%} | Gap: {gap:.2%}")

# Find best
best = max(results, key=lambda x: x['test'])

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")
for r in sorted(results, key=lambda x: x['test'], reverse=True):
    marker = " ← BEST" if r == best else ""
    print(f"{r['desc']:<30} Test={r['test']:.2%}  Gap={r['gap']:.2%}{marker}")
