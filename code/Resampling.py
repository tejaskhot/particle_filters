import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.stats import norm
import pdb
class Resampling:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """

    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """
        self.dice_count = 1

    def multinomial_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        weights = X_bar[:, 3]
        total_num = len(weights)
        weights = weights / np.sum(weights)
        resampled_index = np.random.multinomial(self.dice_count*total_num, weights, size=1)[0, :]
        print('resampled_index : ', resampled_index.shape)
        try:
            X_bar_resampled = X_bar[resampled_index, :]
        except:
            ipdb.set_trace()

        return X_bar_resampled

    def low_variance_sampler(self, X_bar):

        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """

        weights = X_bar[:, 3]/sum(X_bar[:, 3])
        num_particles = X_bar.shape[0]
        X_bar_resampled = np.zeros_like(X_bar)
        # r = np.random.rand() * (1.0/num_particles)
        r = np.random.uniform(low=0., high=(1./num_particles), size=(1,))
        c = weights[0]
        i = 0
        total_weights = np.sum(weights)
        # print('='*75)
        # print('Total particle weights {}'.format(total_weights))
        std = np.std(weights)
        # print('Weights standard deviation {}'.format(std))

        # for m in range(0, num_particles):


        for m in range(0,num_particles):
            u = r + (m-1)*(1.0/num_particles)
            while(u > c):
                i = i + 1
                c = c + weights[i]

            X_bar_resampled[m,:] =  X_bar[i,:]

        # give unifrom weight again
        # X_bar_resampled[:,3] = 1.0/num_particles

        return X_bar_resampled

    # def add_particles(X_bar_resampled):

    #     mean_weight = np.mean(X_bar_resampled[:,3])

    #     if(mean_weight < 0.1):
    #         ## basically all particles are bad


if __name__ == "__main__":
    pass