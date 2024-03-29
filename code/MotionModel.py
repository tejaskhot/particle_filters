import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import pdb

def sample_normal_distribution(b):
    # Slide 19 in  http://ais.informatik.uni-freiburg.de/teaching/ss11/robotics/slides/06-motion-models.pdf?fbclid=IwAR2P0ciQ5bfWzaRVRREwVT1mcHgmrLn8hZyWGcH6bwjpGXO9mpSoJa5K_bc
    r = 0.5 * np.sum(np.random.uniform(-b,b,(12)))

    return r

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

class MotionModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """

    def __init__(self):

        """
        TODO : Initialize Motion Model parameters here
        """
        self.number_particles = 500


    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """

        """
        TODO : Add your code here
        """
        # calibration prams
        alpha_1 = 0.01 # 2 # 5
        alpha_2 = 0.01 # 2 # 7
        alpha_3 = 0.05 # 1 # 7
        alpha_4 = 0.05 # 1 # 5

        # recover the relative motion params - line 2:4
        # ipdb.set_trace()
        delta_rot1 = np.arctan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        delta_trans = np.sqrt((u_t1[1] - u_t0[1])**2 + (u_t1[0] - u_t0[0])**2)
        delta_rot2 = u_t1[2] - u_t0[2] - delta_rot1

        # print(delta_trans, delta_rot2)

        # add the noise into it!
        t1 = alpha_1 * delta_rot1**2 + alpha_2 * delta_trans**2
        t2 = alpha_3 * delta_trans**2 + alpha_4 * (delta_rot1**2 + delta_rot2**2)
        t3 = alpha_1 * delta_rot2**2 + alpha_2 * delta_trans**2

        # t1 = alpha_1 * abs(delta_rot1) + alpha_2 * abs(delta_trans)
        # t2 = alpha_3 * abs(delta_trans) + alpha_4 * (abs(delta_rot1) + abs(delta_rot2))
        # t3 = alpha_1 * abs(delta_rot2) + alpha_2 * abs(delta_trans)

        delta_rot1_cap = delta_rot1  - sample_normal_distribution(t1)
        delta_trans_cap = delta_trans  - sample_normal_distribution(t2)
        delta_rot2_cap = delta_rot2  - sample_normal_distribution(t3)

        x_t1 = np.zeros_like(x_t0)

        x_t1[0] = x_t0[0] + delta_trans_cap * np.cos(x_t0[2] + delta_rot1_cap)
        x_t1[1] = x_t0[1] + delta_trans_cap * np.sin(x_t0[2] + delta_rot1_cap)
        x_t1[2] = x_t0[2] + delta_rot1_cap + delta_rot2_cap

        # print('delta_trans_cap * np.cos(x_t0[2] + delta_rot1_cap) : ', delta_trans_cap * np.cos(x_t0[2] + delta_rot1_cap))
        # print('delta_trans_cap * np.sin(x_t0[2] + delta_rot1_cap): ', delta_trans_cap * np.sin(x_t0[2] + delta_rot1_cap))
        # print('delta_rot1_cap + delta_rot2_cap : ', delta_rot1_cap + delta_rot2_cap)
        # print('='*75)

        return x_t1

    def give_prior(self, x_t1, u_t1, x_t0, u_t0):

        alpha1 = 0.1 # 2 # 5
        alpha2 = 0.1 # 2 # 7
        alpha3 = 0.5 # 1 # 7
        alpha4 = 0.5

        delta_rot1 = np.arctan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        delta_trans = np.sqrt((u_t1[1] - u_t0[1])**2 + (u_t1[0] - u_t0[0])**2)
        delta_rot2 = u_t1[2] - u_t0[2] - delta_rot1

        delta_rot1_cap = np.arctan2(x_t1[1] - x_t0[1], x_t1[0] - x_t0[0]) - x_t0[2]
        delta_trans_cap = np.sqrt((x_t1[1] - x_t0[1])**2 + (x_t1[0] - x_t0[0])**2)
        delta_rot2_cap = x_t1[2] - x_t0[2] - delta_rot1_cap

        p1 = normpdf(delta_rot1 - delta_rot1_cap, 0, alpha1*delta_rot1_cap + alpha2*delta_trans_cap)
        p2 = normpdf(delta_trans - delta_trans_cap, 0, alpha3*delta_trans_cap + alpha4*(delta_rot1_cap + delta_rot2_cap))
        p3 = normpdf(delta_rot2 - delta_rot2_cap, 0, alpha1*delta_rot2_cap + alpha2*delta_trans_cap)

        return p1*p2*p3


if __name__=="__main__":

    # lets move in a rectangle for 500 particles

    x_t0 = np.zeros((500,3), dtype = float)

    time_steps = 300
    u_t0 = np.array([0, 0, 0]).reshape(1,3)
    u_t1 = np.array([10, 0, 0]).reshape(1,3)

    motion_model = MotionModel()

    fig = plt.figure()
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    count = 0

    for i in range(0,time_steps):
        x_t1 = motion_model.update(u_t0, u_t1, x_t0)
        u_t0[0] = u_t1[0]
        u_t1[0,0] = u_t1[0,0] + 20
        print "ut0: " + str(u_t0)
        print "u_t1: " + str(u_t1)
        # pdb.set_trace()
        x_t0 = x_t1

        if(i%10 == 0):
            count = count + 1
            # ax = fig.add_subplot(2,5, count)
            plt.plot(x_t0[:,0], x_t0[:,1],'r.')
            # plt.show()
            plt.hold()

    plt.show()
