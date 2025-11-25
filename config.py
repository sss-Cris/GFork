class Config:
    def __init__(self):
        self.learning_rates = (0.001, 0.0001, 0.00001)
        self.dropouts = (0.3, 0.5)
        self.weight_decays = (0.0, 0.001)
        self.hidden_dims = (64, 96)
        self.mr = {
            "lr": 0.0001,
            "dropout": 0.3,
            "weight_decay": 0.0,
            "hidden_dim": 64,
            "l1_lambda": 0.0,
        }
        self.ohsumed = {
            "lr": 1e-05,
            "dropout": 0.3,
            "weight_decay": 0.001,
            "hidden_dim": 64,
            "l1_lambda": 0.0,
        }
        self.R8 = {
            "lr": 0.000001,
            "dropout": 0.3,
            "weight_decay": 0.001,
            "hidden_dim": 96,
            "l1_lambda": 0.0,
        }
        self.R52 = {
            "lr": 1e-05,
            "dropout": 0.3,
            "weight_decay": 0.0,
            "hidden_dim": 64,
            "l1_lambda": 0.0,
        }
        self.ng20 = {
            "lr": 1e-05,
            "dropout": 0.3,
            "weight_decay": 0.0,
            "hidden_dim": 64,
            "l1_lambda": 0.0,
        }

config = Config()
