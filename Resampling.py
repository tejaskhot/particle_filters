import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.stats import norm

class Resampling:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """

    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """

    def multinomial_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        weights = X_bar[:, 3]
        total_num = len(weights)

        weights = weights / np.sum(weights)
        resampled_index = np.random.multinomial(self.dice_count*total_num, weights, size=1)[0, :]
        X_bar_resampled = X_bar[resampled_index, :]

        return X_bar_resampled

    def low_variance_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        weights = X_bar[:, 3]
        total_weights = np.sum(weights)
        std = np.std(weights)
        print('Total particle weights {}'.format(total_weights))
        print('Weights standard deviation {}'.format(std))

        return X_bar_resampled

    def add_particle(self, X_bar, weight_threshold):
        pass

if __name__ == "__main__":
    pass