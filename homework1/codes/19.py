import perceptron
import numpy as np

X, y = perceptron.preprocess_data(x_0=0)

np.random.seed(42)
rand_seeds = np.random.randint(0, 10000, size=1000)
train_iter_recorder = []
weights_recorder = []
for rand_seed in rand_seeds:
    percp = perceptron.perceptron(X)
    train_iter_recorder.append(percp.fit(X, y, rand_seed))
    weights_recorder.append(percp.W)

print(np.median(train_iter_recorder))