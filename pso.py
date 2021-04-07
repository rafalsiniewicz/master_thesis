from typing import List

from base_class import Base
from particle import Particle

class PSO(Base):
    def __init__(self, name="PSO", dimension=-1, num_features=3, max_iter=100):
        super().__init__(name)
        self.dimension: int = dimension                    # number of dimensions for each particle
        self.num_features: int = num_features              # number of features to select
        self.max_iter: int = max_iter                      # maximum number of iterations
        self.swarm: List[Particle] = []                    # list of particles (swarm)
        self.best_global_position: List[int] = []          # best position of all particles in swarm
        self.best_global_error: float = -1                 # best error of all particles in swarm

    def initialize_swarm(self):
        """
        Initialize swarm of paricles
        :return:
        """
        pass

    def run(self):
        """
        Run PSO algorithm
        :return:
        """
        pass

