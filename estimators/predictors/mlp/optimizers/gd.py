class GradientDescent:
    def __init__(self, eta=1e-5):
        self.eta = eta

    def get_updated_weights(self, current_gradient, current_weights):
        return current_weights - self.eta * current_gradient
