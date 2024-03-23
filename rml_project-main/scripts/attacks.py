import torch
import torch.nn as nn

def pgd(model, X, y, epsilon=0.05, alpha=0.02, num_iter=10, randomize=False, device='cpu'):
    """ Construct PGD adversarial examples for the example (X,y)"""
    x_t = X.clone().requires_grad_(True)

    if randomize:
        random_vector = torch.FloatTensor(x_t.shape).uniform_(-epsilon, epsilon)
        x_t = x_t + random_vector

    delta = 0
    for t in range(num_iter):
        pred = model(x_t.to(device))                         # Generate predictions
        CSE_loss = nn.CrossEntropyLoss()
        loss = CSE_loss(pred, y.to(device))                  # Compute Loss
        loss.backward()

        grad = x_t.grad                                  # Calculate the gradient
        step = alpha*grad.sign()                         # Calculate step update value
        delta += step                                    # Compute the perturbation
        delta = torch.clamp(delta, -epsilon, epsilon)    # Constraint perturbation to bowl Bε(x, ℓ∞)
        x_t.data = X + delta                             # Update the value of x_t

    return delta