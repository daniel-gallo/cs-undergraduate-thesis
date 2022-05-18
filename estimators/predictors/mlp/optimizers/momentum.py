from estimators.predictors.mlp.optimizers.gd import GradientDescent


class Momentum(GradientDescent):
    def __init__(self, beta=0.9, eta=1e-5):
        super().__init__(eta)
        self.beta = beta

        self.momentum = 0

    def get_updated_weights(self, current_gradient, current_weights):
        self.momentum = self.beta * self.momentum + current_gradient
        return current_weights - self.eta * self.momentum
