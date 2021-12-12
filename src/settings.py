"""
File to store settings and constants
"""


class Config:
    def __init__(self) -> None:
        self.grid_size = {"x": 5, "y": 10}
        self.n_clusters = 2
        self.pca_components = 10
        self.random_state = 88
        self.turnover_zone_id = 99998
        self.score_zone_id = 99999


def get_config() -> Config:
    return Config()
