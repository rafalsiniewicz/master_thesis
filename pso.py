from typing import List

from base_class import Base
from particle import Particle

class PSO(Base):
    def __init__(self, name="PSO", dimension=-1, num_features=3, max_iter=100, num_particles=10):
        super().__init__(name)
        self.dimension: int = dimension                    # number of dimensions for each particle
        self.num_features: int = num_features              # number of features to select
        self.max_iter: int = max_iter                      # maximum number of iterations
        self.swarm: List[Particle] = []                    # list of particles (swarm)
        self.best_global_position: List[int] = []          # best position of all particles in swarm
        self.best_global_error: float = -1                 # best error of all particles in swarm
        self.num_particles: int = num_particles


    def set_particles_parameters(self, w=0.5, c1=1, c2=2):
        """
        Set parameters for all particles.
        :param w:
        :param c1:
        :param c2:
        :return:
        """
        Particle.dimension = self.dimension
        Particle.num_features = self.num_features
        Particle.w = w
        Particle.c1 = c1
        Particle.c2 = c2


    def initialize_swarm(self):
        """
        Initialize swarm of particles
        :return self.swarm (list[Particle]):                swarm of particles
        """
        self.set_particles_parameters()
        self.swarm = [Particle() for d in range(self.num_particles)]
        return self.swarm

    def run(self):
        """
        Run PSO algorithm.
        :return:
        """
        pass



if __name__ == "__main__":
    pso = PSO(dimension=10)
    pso.initialize_swarm()
    for i in pso.swarm:
        print(i)

