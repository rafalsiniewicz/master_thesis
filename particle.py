from random import uniform

from base_class import Base


class Particle(Base):
    dimension = -1  # number of dimensions for particles

    def __init__(self, name="PARTICLE", position=None, velocity=None):
        super().__init__(name)
        if position:  # TODO sprawdzic czy typy sa odpowiednie, tzn. tutaj lista wartosci binarnych
            self.position = position
        else:
            self.position = []  # wylosowac jakis wektor binarny o wymiarach dimension x 1
        if velocity:  # TODO sprawdzic czy typy sa odpowiednie, tzn. tutaj lista wartosci float
            self.velocity = velocity
        else:
            self.velocity = [uniform(-1, 1) for d in range(Particle.dimension)]
        self.best_position = []
        self.best_error = -1
        self.error = -1



    def update_velocity(self):
        pass


    def update_position(self):
        pass

    def evaluate(self):
        pass


