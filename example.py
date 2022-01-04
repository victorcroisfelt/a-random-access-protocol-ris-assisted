from environment.box import Box
import numpy as np

# General parameter
ell = 100
ell0 = 10

K = 5

# Initialize the environment
box = Box(ell, ell0)
box.place_bs(pos=np.array([30, np.deg2rad(90 + 45)]))
# box.place_bs()
box.place_ue(K)     # place K users
box.place_ris()
box.plot_scenario()

# box.build_chan_gain()   # still work in progress

