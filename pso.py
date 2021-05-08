from base_class import *
from particle import Particle
from document import Document
from corpus import Corpus
from copy import deepcopy


class PSO(Base):
    selected_features = []  # list of all selected features for all documents
    def __init__(self, name="PSO", num_features_select=3, max_iter=100, num_particles=10, features=[]):
        super().__init__(name)
        self.num_features_select: int = num_features_select  # number of features to select
        self.features: List[float] = features  # features for all particles (vector of tf_idf values)
        self.max_iter: int = max_iter  # maximum number of iterations
        self.swarm: List[Particle] = []  # list of particles (swarm)
        self.best_global_position: List[int] = []  # best position of all particles in swarm
        self.best_global_error: float = -1  # best error of all particles in swarm
        self.num_particles: int = num_particles
        self.best_particle = None

    def set_particles_parameters(self, w=0.5, c1=1, c2=2):
        """
        Set parameters for all particles.
        :param w:
        :param c1:
        :param c2:
        :return:
        """
        Particle.num_features_select = self.num_features_select
        Particle.features = self.features
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

    def get_values_for_selected_features(self):
        """
        Leave only selected features (values) in vector. The rest is set to:
         a) -1: in case the feature doesn't appear in document or
         b) 0: in case the feature appear in document, but is not selected
        :return:
        """
        selected_features = []
        for i in range(len(self.best_global_position)):
            if self.best_global_position[i] == 1:
                selected_features.append(self.features[i])
            else:
                selected_features.append(float(self.best_global_position[i]))

        return selected_features


    def run(self, get_values_for_selected_features=True):
        """
        Run PSO algorithm.
        :return self.best_particle:         # best particle object from whole swarm (best solution)
        """

        # begin optimization loop
        i = 0
        while i < self.max_iter:
            # print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(self.num_particles):
                self.swarm[j].evaluate()
                # determine if current particle is the best (globally)
                if self.swarm[j].error > self.best_global_error or self.best_global_error == -1:
                    self.best_global_position = list(self.swarm[j].position)
                    self.best_global_error = float(self.swarm[j].error)
                    self.best_particle = deepcopy(self.swarm[j])

            # cycle through swarm and update velocities and position
            for j in range(self.num_particles):
                self.swarm[j].update_velocity(self.best_global_position)
                self.swarm[j].update_position()

            i += 1

        # print final results
        print('FINAL:\n')
        print("self.best_global_position", self.best_global_position)
        print("self.best_global_error", self.best_global_error)
        print("self.best_particle:", self.best_particle)

        if get_values_for_selected_features:
            # take float tf_idf values for selected features (values 1 in particle vector)
            sf = self.get_values_for_selected_features()
            # add selected features to list of all selected features for all documents
            PSO.selected_features.append(sf)
        else:
            # add selected features to list of all selected features for all documents
            PSO.selected_features.append(self.best_global_position)

        return self.best_particle


if __name__ == "__main__":
    c = Corpus()
    c.read_documents()
    c.get_vocabulary()
    c.tf_idf()
    for k in range(3):
        pso = PSO(features=c.documents[k].features, max_iter=100)
        pso.set_particles_parameters()
        pso.initialize_swarm()
        # for i in pso.swarm:
        #     print(i)

        pso.run()

    for i in PSO.selected_features:
        print(i)
        # print(len(i))
