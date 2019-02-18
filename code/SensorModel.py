import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm
from numpy import linalg as LA
import scipy.stats
from multiprocessing.dummy import Pool
import ipdb

from MapReader import MapReader

def init_particles_freespace(num_particles, occupancy_map):
    # initialize [x, y, theta] positions in world frame for all particles
    # (in the unoccupied/empty cells of the map)
    num_empty = np.sum(occupancy_map == 0)
    y_empty, x_empty = np.where(occupancy_map == 0)
    randidx = np.random.randint(num_empty, size=(num_particles, 1))
    # create gaussian particles at random locations
    x0_vals = (x_empty[randidx] + np.random.uniform(size=(num_particles, 1))) * 10
    y0_vals = (y_empty[randidx] + np.random.uniform(size=(num_particles, 1))) * 10
    theta0_vals = np.random.uniform(-np.pi, np.pi, (num_particles, 1))
    # all start with same weight
    w = np.ones((num_particles, 1)) / num_particles

    X_bar_init = np.hstack((x0_vals, y0_vals, theta0_vals, w) )

    return X_bar_init

class SensorModel:

    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map):
        self.map = occupancy_map

        self.w_short = 10
        self.w_max = 0.5
        self.w_rand = 9.5
        self.w_hit = 80
        self.sigma_hit = 80.0
        self.lambda_short = 0.02

        self.z_max = 1000

        self.subsample_step_size = 20

        self.obstacle_threshold = 0.1

        self.gaussian_const = self.sigma_hit * math.sqrt(2 * math.pi)

        self.visualization = False
        self.plot_measurement = False
        self.sensor_model_distribution_vis = False
        self.debug_msg = False

        self.pool = Pool(processes=4)

        self.weight_sum = self.w_short + self.w_max + self.w_rand + self.w_hit

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        x_t1 = [x_t1[0]/10, x_t1[1]/10, x_t1[2]]
        if self.map[int(x_t1[1])][int(x_t1[0])] > self.obstacle_threshold:
            return 0

        if self.visualization:
            self.visualize_map()

        heading = x_t1[2]
        theta_start = heading - math.pi / 2
        # ipdb.set_trace()
        z_true = np.zeros(180)
        z_measured = np.array(z_t1_arr)

        for i in range(0, 180, self.subsample_step_size):
            theta = theta_start + i / 180.0 * math.pi

            for j in range(0, self.z_max / 10):
                x_idx = int(round(max(min(x_t1[0] + j * math.cos(theta), len(self.map) - 1), 0)))
                y_idx = int(round(max(min(x_t1[1] + j * math.sin(theta), len(self.map[0]) - 1), 0)))
                # print(x_idx, y_idx)
                if self.map[y_idx][x_idx] > self.obstacle_threshold:
                    z_true[i] = j * 10

                    if self.visualization:
                        plt.plot([x_t1[0], x_idx], [x_t1[1], y_idx], 'r-')
                        if self.plot_measurement:
                            plt.plot([x_t1[0], x_t1[0] + z_t1_arr[i] * math.cos(theta) / 10], [x_t1[1], x_t1[1] + z_t1_arr[i] * math.sin(theta) / 10], 'b-')
                    break

        z_true = z_true[::self.subsample_step_size]
        z_measured = z_measured[::self.subsample_step_size]

        p_hit = self.p_hit(z_true, z_measured)
        p_short = self.p_short(z_true, z_measured)
        p_max = self.p_max(z_measured)
        p_rand = self.p_rand(z_measured)
        q = np.log(self.w_hit * p_hit + self.w_short * p_short + self.w_max * p_max + self.w_rand * p_rand)
        q = np.exp(q.mean())
        #  q = np.exp(np.median(q))

        if self.debug_msg:
            print(z_true)
            print(z_measured)
            print("probability: %f" % q)

        if self.sensor_model_distribution_vis:
            z_true = np.ones((200,)) * 500
            x = np.linspace(0, self.z_max, 200)
            y = self.w_hit * self.p_hit(z_true, x) + self.w_rand * self.p_rand(x) + self.w_short * self.p_short(z_true, x) + self.w_max * self.p_max(x)
            plt.plot(x, y, 'k-')
            # plt.show()
            plt.pause(200)
            # plt.pause(0.00001)

        if self.visualization:
            plt.plot([x_t1[0]], [x_t1[1]], 'go')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.draw()
            plt.show()
            plt.pause(1)
            # plt.pause(0.00001)
            plt.close()

        return q

    def p_hit(self, z_true, z_measured):
        p_hit = np.exp(-(z_true - z_measured)**2 / (2.0 * self.sigma_hit**2)) / self.gaussian_const
        return p_hit

    def p_short(self, z_true, z_measured):
        within_range_idx = np.where(z_measured <= z_true)
        p_short = np.zeros(len(z_true))
        p_short[within_range_idx] = 1 / (1 - np.exp(-self.lambda_short * z_true[within_range_idx])) * self.lambda_short * np.exp(-self.lambda_short * z_measured[within_range_idx])
        return p_short

    def p_max(self, z_measured):
        max_range_idx = np.where(z_measured >= self.z_max)
        p_max = np.zeros(len(z_measured))
        p_max[max_range_idx] = 1
        return p_max

    def p_rand(self, z_measured):
        within_range_idx = np.where(z_measured < self.z_max)
        p_rand = np.zeros(len(z_measured))
        p_rand[within_range_idx] = 1.0 / self.z_max
        return p_rand

    def visualize_map(self):
        fig = plt.figure()
        mng = plt.get_current_fig_manager();
        mng.resize(*mng.window.maxsize())
        plt.ion()
        plt.imshow(self.map, cmap='Greys')
        plt.axis([300, 700, 0, 800])
        # plt.draw()
        # plt.pause(200)

if __name__=='__main__':
    src_path_map = '../data/map/wean.dat'
    src_path_log = '../data/log/robotdata1.log'

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')
    sensor_model = SensorModel(occupancy_map)

    for time_idx, line in enumerate(logfile):
        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        meas_type = line[0] # L : laser scan measurement, O : odometry measurement
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ') # convert measurement values from string to double

        odometry_robot = meas_vals[0:3] # odometry reading [x, y, theta] in odometry frame
        time_stamp = meas_vals[-1]

        if ((time_stamp <= 0.0) | (meas_type == "O")): # ignore pure odometry measurements for now (faster debugging)
            continue

        if (meas_type == "L"):
             odometry_laser = meas_vals[3:6] # [x, y, theta] coordinates of laser in odometry frame
             ranges = meas_vals[6:-1] # 180 range measurement values from single laser scan

        print("Processing time step " + str(time_idx) + " at time " + str(time_stamp) + "s")

        X = init_particles_freespace(5, occupancy_map)
        # laser_pos_test = [650, 145, -math.pi / 2]
        laser_pos_test = [4000, 4000, np.pi]
        q = sensor_model.beam_range_finder_model(ranges, laser_pos_test)
        print(q)
        # break
        # if time_idx == 9:
        #     break