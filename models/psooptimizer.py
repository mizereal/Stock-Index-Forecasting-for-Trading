BIG_SCORE = 1.e7  # type: float

import tensorflow.keras as keras
from models.particle import Particle
from utils.progressbar import ProgressBar

class Optimizer:
    def __init__(self, model, loss,
                 n=10,
                 inertia_weight=0.7298,
                 local_rate=1.49618,
                 global_rate=1.49618):

        self.n_particles = n
        self.structure = model.to_json()
        self.particles = [None] * n
        self.loss = loss
        self.length = len(model.get_weights())

        params = {'w': inertia_weight, 'local_acc': local_rate, 'global_acc': global_rate}

        for i in range(n-1):
            m = keras.models.model_from_json(self.structure)
            m.compile(loss=loss,optimizer='sgd')
            self.particles[i] = Particle(m, params)

        self.particles[n-1] = Particle(model, params)

        self.global_best_weights = None
        self.global_best_score = BIG_SCORE

    def fit(self, x, y, steps=0, batch_size=32):
        batch_size = 32 if batch_size == None else batch_size
        num_batches = x.shape[0] // batch_size

        for i, p in enumerate(self.particles):
            local_score = p.get_score(x, y)

            if local_score < self.global_best_score:
                self.global_best_score = local_score
                self.global_best_weights = p.get_best_weights()

        print("PSO -- Initial best score {:0.4f}".format(self.global_best_score))

        bar = ProgressBar(steps, updates=20)

        for i in range(steps):
            for j in range(num_batches):
                x_ = x[j*batch_size:(j+1)*batch_size,:]
                y_ = y[j*batch_size:(j+1)*batch_size]

                for p in self.particles:
                    local_score = p.step(x_, y_, self.global_best_weights)

                    if local_score < self.global_best_score:
                        self.global_best_score = local_score
                        self.global_best_weights = p.get_best_weights()
                        
            bar.update(i)

        bar.done()

    def get_best_model(self):
        best_model = keras.models.model_from_json(self.structure)
        best_model.set_weights(self.global_best_weights)
        best_model.compile(loss=self.loss,optimizer='adam')
        return best_model
