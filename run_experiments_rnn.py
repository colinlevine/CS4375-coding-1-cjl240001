import subprocess

# Run experiments with varying hidden dimensions (doubling each time)
hidden_dim = 10
while hidden_dim <= 5120:
    print(f"\n{'='*50}")
    print(f"Running experiment with hidden_dim = {hidden_dim}")
    print(f"{'='*50}\n")

    # Run rnn.py with the current hidden dimension
    subprocess.run(['python', 'rnn.py', '-hd', str(hidden_dim)])

    print(f"\nCompleted experiment with hidden_dim = {hidden_dim}")

    hidden_dim *= 2

print("\n" + "="*50)
print("All experiments completed!")
print("="*50)
