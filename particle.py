from random import uniform, randint, random
from numpy import exp
from base_class import *
from statistics import mean


class Particle(Base):
    w = 0.5  # constant inertia weight (how much to weigh the previous velocity)
    c1 = 1  # cognitive constant
    c2 = 2  # social constant
    num_features_select = 0  # number of particle's features
    id = 0  # particle id
    features = []  # list of particle's features

    def __init__(self, name="PARTICLE", position=None, velocity=None):
        super().__init__(name)
        self.id = Particle.id
        Particle.id += 1
        if Particle.num_features_select > len(Particle.features):
            raise ValueError("Number of features to select must be lower or equal to vector dimension")
        if position and isinstance(position, list) and all(element in [-1, 0, 1] for element in position):
            self.position = position
        elif position and (not isinstance(position, list) or not all(element in [0, 1] for element in position)):
            raise TypeError("Position of particles must be a list of binary values")
        else:
            """
            Position values explanation:
            self.position[i] = -1:  word doesnt exist in the document (tf_idf = 0)
            self.position[i] = 0:   word exist in the document but is not selected
            self.position[i] = 1:   word exist in the document and is selected as a feature
            """
            self.position = [-1] * len(Particle.features)  # initialize all positions as -1's
            for i in range(len(Particle.features)):  # initialize position with tf_df != 0 for feature as 0
                if Particle.features[i] != 0:
                    self.position[i] = 0
            # randomly choose "num_features_select" 1's, so randomly choose proper number of selected features (from 0's)
            selected_features = 0
            while selected_features < Particle.num_features_select:
                selected = randint(0, len(self.position) - 1)
                if self.position[selected] == 0:
                    self.position[selected] = 1
                    selected_features += 1

        if velocity and isinstance(position,
                                   list):  # TODO sprawdzic czy typy sa odpowiednie, tzn. tutaj lista wartosci float/int
            self.velocity = velocity
        else:
            self.velocity = [uniform(-1, 1) for d in range(len(Particle.features))]
        self.best_position = []
        self.best_error = -1
        self.error = -1

    def __str__(self):
        return "\n### Particle ###\nid = {}\nposition={}\nvelocity={}\nbest_position={}\nbest_error={}\nerror={}\n".format(
            self.id, self.position, self.velocity, self.best_position, self.best_error, self.error)

    def update_velocity(self, best_global_position):
        """
        Update particle velocity from equation:
        v^{i}_{k+1} = w_{k}*v^{i}_{k} + c_{1}*r_{1}*(p^{i}_{k} - x^{i}_{k}) + c_{2}*r_{2}*(p^{g}_{k} - x^{i}_{k})
        :param best_global_position:
        :return:


        """
        # self.log_debug("Updating particle's velocity")
        for i in range(len(Particle.features)):
            r1 = random()
            r2 = random()

            vel_cognitive = self.c1 * r1 * (self.best_position[i] - self.position[i])
            vel_social = self.c2 * r2 * (best_global_position[i] - self.position[i])
            self.velocity[i] = self.w * self.velocity[i] + vel_cognitive + vel_social

    def update_position(self):
        """
        Update particle position.
        :return:
        """
        # self.log_debug("Updating particle's position")

        for i in range(len(Particle.features)):
            if self.position[i] != -1:
                if random() < self.sigmoid(self.velocity[i]):
                    self.position[i] = 1
                else:
                    self.position[i] = 0

    def evaluate(self, function='MAD'):
        self.error = self.cost_function(function=function)

        # check to see if the current position is an individual best
        if self.error > self.best_error or self.best_error == -1:
            self.best_position = self.position
            self.best_error = self.error

    @staticmethod
    def sigmoid(x):
        z = exp(-x)
        sig = 1 / (1 + z)

        return sig

    def cost_function(self, function='MAD'):
        """
        Calculate cost funtion value for particle.
        :return:
        """
        if function not in ['MAD', 'MAX']:
            raise ValueError("Function must be MAD or MAX")
        result = 0
        solution = []  # selected features values
        for i in range(len(self.position)):
            if self.position[i] == 1:
                solution.append(Particle.features[i])
        if function == 'MAD':
            average_solution = mean(solution)
            for i in solution:
                result += abs(i - average_solution)

            result = result / len(solution)
            # print("result: ", result)
        elif function == 'MAX':
            for i in solution:
                result += i
            # print("result: ", result)

        return result
