import numpy as np
import sys
import pdb

from MapReader import MapReader
from MotionModel import MotionModel
from SensorModel import SensorModel
from Resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib import figure as fig
import time

def visualize_map(occupancy_map):
    fig = plt.figure()
    # plt.switch_backend('TkAgg')
    mng = plt.get_current_fig_manager();  # mng.resize(*mng.window.maxsize())
    plt.ion(); plt.imshow(occupancy_map, cmap='Greys'); plt.axis([0, 800, 0, 800]);


def visualize_timestep(X_bar, tstep):
    x_locs = X_bar[:,0]/10.0
    y_locs = X_bar[:,1]/10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o')
    plt.pause(0.00001)
    scat.remove()

def init_particles_random(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles
    y0_vals = np.random.uniform( 0, 7000, (num_particles, 1) )
    x0_vals = np.random.uniform( 3000, 7000, (num_particles, 1) )
    theta0_vals = np.random.uniform( -3.14, 3.14, (num_particles, 1) )

    # initialize weights for all particles
    w0_vals = np.ones( (num_particles,1), dtype=np.float64)
    w0_vals = w0_vals / num_particles

    X_bar_init = np.hstack((x0_vals,y0_vals,theta0_vals,w0_vals))

    return X_bar_init

def init_particles_freespace(num_particles, occupancy_map):

    # initialize [x, y, theta] positions in world_frame for all particles

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

def main():

    """
    Description of variables used
    u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
    u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
    x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
    x_t1 : particle state belief [x, y, theta] at time t [world_frame]
    X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
    z_t : array of 180 range measurements for each laser scan
    """

    """
    Initialize Parameters
    """
    src_path_map = '../data/map/wean.dat'
    src_path_log = '../data/log/robotdata1.log'

    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')

    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()

    num_particles = 100
    time_period = 10
    # X_bar = init_particles_random(num_particles, occupancy_map)
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    vis_flag = 1
    vis_type = 'mean' # {mean, max}

    """
    Monte Carlo Localization Algorithm : Main Loop
    """
    if vis_flag:
        visualize_map(occupancy_map)

    first_time_idx = True
    count = 0
    for time_idx, line in enumerate(logfile):
        # if time_idx % 9 != 0: continue
        
        # Read a single 'line' from the log file (can be either odometry or laser measurement)
        meas_type = line[0] # L : laser scan measurement, O : odometry measurement
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ') # convert measurement values from string to double

        odometry_robot = meas_vals[0:3] # odometry reading [x, y, theta] in odometry frame
        time_stamp = meas_vals[-1]

        if ((time_stamp <= 0.0) | (meas_type == "O")): # ignore pure odometry measurements for now (faster debugging)
            continue

        count = count + 1

        if (meas_type == "L"):
             odometry_laser = meas_vals[3:6] # [x, y, theta] coordinates of laser in odometry frame
             ranges = meas_vals[6:-1] # 180 range measurement values from single laser scan

        # print "Processing time step " + str(time_idx) + " at time " + str(time_stamp) + "s"

        if (first_time_idx):
            u_t0 = odometry_robot
            first_time_idx = False
            continue

        X_bar_new = np.zeros( (num_particles,4), dtype=np.float64)
        u_t1 = odometry_robot

        for m in range(0, num_particles):

            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)

            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges
                w_t = sensor_model.beam_range_finder_model(z_t, x_t1)
                # w_t = 1/num_particles
                # X_bar_new[m,:] = np.hstack((x_t1, w_t))
                new_wt = X_bar[m,3] * motion_model.give_prior(x_t1,u_t1,x_t0,u_t0)
                X_bar_new[m,:] = np.hstack((x_t1, new_wt))
            else:
                X_bar_new[m,:] = np.hstack((x_t1, X_bar[m,3]))

        if (vis_type == 'max'):
            best_particle_idx = np.argmax(X_bar_new, axis=0)[-1]
            vis_particle = X_bar_new[best_particle_idx][:-1]
        elif (vis_type == 'mean'):
            # ipdb.set_trace()
            X_weighted = X_bar_new[:,:3] * X_bar_new[:,3:4]
            X_mean = np.sum(X_weighted, axis=0)
            vis_particle = X_mean/sum(X_bar_new[:,3:4])

        # print(X_bar_new[:,-1].T)
        sensor_model.visualization = True
        sensor_model.plot_measurement = True
        # sensor_model.beam_range_finder_model(ranges, vis_particle)
        sensor_model.visualization = False
        sensor_model.plot_measurement = False

        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
        #if(np.mean(x_t1 - x_t0) > 0.2):
        X_bar = resampler.low_variance_sampler(X_bar)
        add_particles = num_particles/5
        # time_period = 10
        
        if(count%time_period == 0 or sum(X_bar[:,-1])==0):
            X_bar_re_init = init_particles_freespace(add_particles,occupancy_map) 
            X_bar[:,-1] = 1.0/(num_particles+add_particles)
            X_bar_re_init[:,-1] = 1.0/(num_particles+add_particles)
            X_bar = np.concatenate((X_bar, X_bar_re_init),axis = 0)
            num_particles = X_bar.shape[0]
            print num_particles
        
        if(count%100 == 0):
            time_period = time_period * 5
        
        # X_bar = resampler.multinomial_sampler(X_bar)
        # check if importance too low
        # thres = 1e-29
        # indices = np.where(X_bar[:,3] > thres)[0]
        # print(X_bar.shape[0] - indices.shape[0])
        # temp = init_particles_freespace(X_bar.shape[0] - indices.shape[0], occupancy_map)
        # X_bar = np.concatenate((X_bar[indices], temp), axis = 0)
        # X_bar[:,-1] = 1.0/num_particles

        if vis_flag:
            visualize_timestep(X_bar, time_idx)

if __name__=="__main__":
    main()
