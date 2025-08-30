import numpy as np
import matplotlib.pyplot as plt

# Adagrad algorithm simulation
def adagrad_example():
    # Parameters
    initial_lr = 0.1  # Initial learning rate
    epsilon = 1e-8    # Small constant to avoid division by zero
    num_iterations = 100
    
    # Simulate some gradients (let's say they're relatively constant)
    # In practice, gradients vary, but this shows the monotonic decay clearly
    gradients = np.random.normal(0.5, 0.1, num_iterations)  # Mean 0.5, std 0.1
    
    # Initialize
    accumulated_gradients = 0  # Sum of squared gradients
    effective_learning_rates = []
    
    print("Adagrad Learning Rate Evolution:")
    print("Iteration | Gradient | Accumulated G² | Effective LR")
    print("-" * 55)
    
    for i in range(num_iterations):
        gradient = gradients[i]
        
        # Update accumulated squared gradients
        accumulated_gradients += gradient ** 2
        
        # Calculate effective learning rate
        effective_lr = initial_lr / (np.sqrt(accumulated_gradients) + epsilon)
        effective_learning_rates.append(effective_lr)
        
        # Print first few and some later iterations
        if i < 5 or i % 20 == 0:
            print(f"{i+1:9d} | {gradient:8.3f} | {accumulated_gradients:13.3f} | {effective_lr:11.6f}")
    
    return effective_learning_rates, gradients

# Run the simulation
learning_rates, gradients = adagrad_example()

# Plot the results
plt.figure(figsize=(12, 8))

# Plot 1: Learning rate decay
plt.subplot(2, 2, 1)
plt.plot(range(1, len(learning_rates) + 1), learning_rates)
plt.title('Adagrad: Effective Learning Rate Over Time')
plt.xlabel('Iteration')
plt.ylabel('Effective Learning Rate')
plt.grid(True)

# Plot 2: Accumulated gradients
accumulated_grads = np.cumsum(np.array(gradients) ** 2)
plt.subplot(2, 2, 2)
plt.plot(range(1, len(accumulated_grads) + 1), accumulated_grads)
plt.title('Accumulated Squared Gradients')
plt.xlabel('Iteration')
plt.ylabel('Sum of G²')
plt.grid(True)

# Plot 3: Gradients over time
plt.subplot(2, 2, 3)
plt.plot(range(1, len(gradients) + 1), gradients)
plt.title('Gradients Over Time')
plt.xlabel('Iteration')
plt.ylabel('Gradient Value')
plt.grid(True)

# Plot 4: Learning rate on log scale to show the decay more clearly
plt.subplot(2, 2, 4)
plt.semilogy(range(1, len(learning_rates) + 1), learning_rates)
plt.title('Learning Rate Decay (Log Scale)')
plt.xlabel('Iteration')
plt.ylabel('Effective Learning Rate (log)')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\nInitial learning rate: {0.1}")
print(f"Final learning rate: {learning_rates[-1]:.8f}")
print(f"Reduction factor: {0.1 / learning_rates[-1]:.2f}x smaller")

# Show the mathematical formula
print("\nAdagrad Formula:")
print("effective_lr = initial_lr / (√(Σ(g²)) + ε)")
print("where g² are squared gradients accumulated over all previous steps")
print("\nKey insight: The denominator √(Σ(g²)) only grows larger, so effective_lr only decreases!")
