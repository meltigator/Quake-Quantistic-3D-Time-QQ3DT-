# Lorenz attractor
def lorenz_model(state, dt, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return state + dt * np.array([dx, dy, dz])

# RNN simple predictor
class SimpleRNNPredictor:
    def __init__(self, hidden_size=8):
        self.W = np.random.randn(hidden_size, 2) * 0.1
        self.U = np.random.randn(2, hidden_size) * 0.1
        
    def __call__(self, state, dt):
        # Forward pass semplice
        h = np.tanh(self.U @ state)
        prediction = self.W @ h
        return state + dt * prediction