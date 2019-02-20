import numpy as np
import sys
import ipdb

from MapReader import MapReader
from MotionModel import MotionModel
from SensorModel import SensorModel
from Resampling import Resampling

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
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

# def main():

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
src_path_log = '../data/log/robotdata2.log'

map_obj = MapReader(src_path_map)
occupancy_map = map_obj.get_map()
# logfile = open(src_path_log, 'r')
with open(src_path_log, 'r') as f:
    logs = f.readlines()

motion_model = MotionModel()
sensor_model = SensorModel(occupancy_map)
resampler = Resampling()

num_particles = 500
# X_bar = init_particles_random(num_particles, occupancy_map)
X_bar = init_particles_freespace(num_particles, occupancy_map)
vis_flag = 1
vis_type = 'mean' # {mean, max}

"""
Monte Carlo Localization Algorithm : Main Loop
"""
if vis_flag:
    visualize_map(occupancy_map)

# initialize plot to save the video of the full sequence
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(figsize=(8,8))
plt.imshow(occupancy_map) # optional : cmap='viridis'
plt.axis([0,800,0,800])
scat = plt.scatter(X_bar[:,0]/10, X_bar[:,1]/10, s=10.0, c='r', marker='o') # optional: marker='o'
plt.title('Timestep 0') # this will be updated as the log files are read


# read the file again since previously we only counted number of lines
logfile = open(src_path_log, 'r')
# use time step 0 ie first line of logs for initializing and ignore afterwards
line = next(logfile)
# Read a single 'line' from the log file (can be either odometry or laser measurement)
meas_type = line[0] # L : laser scan measurement, O : odometry measurement
meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ') # convert measurement values from string to double
odometry_robot = meas_vals[0:3] # odometry reading [x, y, theta] in odometry frame
time_stamp = meas_vals[-1]
odometry_laser = meas_vals[3:6] # [x, y, theta] coordinates of laser in odometry frame
ranges = meas_vals[6:-1] # 180 range measurement values from single laser scan
u_t0 = odometry_robot
count = 0
time_period = 10

# animation function.  This is called sequentially
def optimize(time_idx):
    # need access to global variables here
    global X_bar, u_t0, count, time_period, num_particles

    # Read a single 'line' from the log file (can be either odometry or laser measurement)
    line = next(logfile)
    meas_type = line[0] # L : laser scan measurement, O : odometry measurement
    meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ') # convert measurement values from string to double
    odometry_robot = meas_vals[0:3] # odometry reading [x, y, theta] in odometry frame
    time_stamp = meas_vals[-1]

    # this return here does not work and animation part complains so pretend to not do anything if this happens
    # if ((time_stamp <= 0.0) | (meas_type == "O")): # ignore pure odometry measurements for now (faster debugging)
    #     return

    if (meas_type == "L"):
         odometry_laser = meas_vals[3:6] # [x, y, theta] coordinates of laser in odometry frame
         ranges = meas_vals[6:-1] # 180 range measurement values from single laser scan

    print "Processing time step " + str(time_idx+1) + " at time " + str(time_stamp) + "s"

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
            X_bar_new[m,:] = np.hstack((x_t1, w_t))

    u_t0 = u_t1
    if (meas_type == "L"):
        X_bar = X_bar_new
        """
        RESAMPLING
        """
        X_bar = resampler.low_variance_sampler(X_bar)
        # time_period = 10

        if(count%time_period == 0 or sum(X_bar[:,-1])==0):
            add_particles = num_particles/5
            X_sorted = X_bar[X_bar[:,-1].argsort()][add_particles:]
            X_bar_re_init = init_particles_freespace(add_particles,occupancy_map)
            X_bar = np.concatenate((X_sorted, X_bar_re_init),axis=0)
            X_bar[:,-1] = 1.0/(len(X_bar))
            # X_bar_re_init[:,-1] = 1.0/(num_particles+add_particles)
            # X_bar = np.concatenate((X_bar, X_bar_re_init),axis = 0)
            num_particles = X_bar.shape[0]
            print num_particles

        # if(count%100 == 0):
        #     time_period = time_period * 5

    # update plot in place with the newly updated values
    scat.set_offsets(X_bar[:, 0:2] / 10.0)
    plt.title('Time step %d' % (time_idx + 1))

# call the animator.  blit=True means only re-draw the parts that have changed.
# animation = FuncAnimation(fig=fig, func=optimize, frames=len(logs), interval=30, blit=True)
animation = FuncAnimation(fig, optimize, len(logs), interval=40)
animation.save('../data/temp.gif', dpi=40, writer='imagemagick')


# if __name__=="__main__":
#     main()
