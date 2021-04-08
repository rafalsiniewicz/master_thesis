from random import uniform, shuffle, random
from numpy import exp
from base_class import Base


class Particle(Base):
    dimension = 0   # number of dimensions for particles
    w = 0.5         # constant inertia weight (how much to weigh the previous velocity)
    c1 = 1          # cognitive constant
    c2 = 2          # social constant
    num_features = 0
    id = 0

    def __init__(self, name="PARTICLE", position=None, velocity=None):
        super().__init__(name)
        self.id = Particle.id
        Particle.id += 1
        if Particle.num_features > Particle.dimension:
            raise ValueError("Number of features to select must be lower or equal to vector dimension")
        if position and isinstance(position, list) and all(element in [0, 1] for element in position):
            self.position = position
        elif position and (not isinstance(position, list) or not all(element in [0, 1] for element in position)):
            raise TypeError("Position of particles must be a list of binary values")
        else:
            # initialize position vector as binary vector with exact "num_features" of 1's and the rest numbers are 0
            selected = [1] * Particle.num_features  # list of 1's (selected features)
            not_selected = [0] * (Particle.dimension - Particle.num_features)  # list of 0's ( not selected features)
            self.position = selected + not_selected
            shuffle(self.position)
        if velocity and isinstance(position,
                                   list):  # TODO sprawdzic czy typy sa odpowiednie, tzn. tutaj lista wartosci float/int
            self.velocity = velocity
        else:
            self.velocity = [uniform(-1, 1) for d in range(Particle.dimension)]
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
        self.log_debug("Updating particle's velocity")
        for i in range(Particle.dimension):
            r1 = random()
            r2 = random()

            vel_cognitive = Particle.c1 * r1 * (self.best_position[i] - self.position[i])
            vel_social = Particle.c2 * r2 * (best_global_position[i] - self.position[i])
            self.velocity[i] = Particle.w * self.velocity[i] + vel_cognitive + vel_social

    def update_position(self):
        """
        Update particle position.
        :return:
        """
        self.log_debug("Updating particle's position")
        for i in range(Particle.dimension):
            if random() < self.sigmoid(self.velocity[i]):
                self.position[i] = 1
            else:
                self.position[i] = 0

    def evaluate(self):
        self.error = self.cost_function()

        # check to see if the current position is an individual best
        if self.error < self.best_error or self.best_error == -1:
            self.best_position = self.position
            self.best_error = self.error

    @staticmethod
    def sigmoid(x):
        z = exp(-x)
        sig = 1 / (1 + z)

        return sig

    def cost_function(self):
        """
        Calculate cost funtion value for particle.
        :return:
        """
        MAD = 0
        # TODO implement MAD as a cost function
        pass
        return MAD
