# simulation of Amari-Hopfield network
# notation from Hopfield (PNAS, 1982)
# Tatsuo Okubo
# 2024/12/14

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import os
import utils

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# parameters
size = (16, 16)  # size of the image
N = size[0] * size[1]  # number of neurons
L = 2000  # number of time steps
visualize = True  # whether to visualize patterns to memorize

# load images
patterns = np.load("data/images.npz")
images = {name: patterns[name] for name in patterns.files}  # Convert to a dictionary
print("Patterns in the file:", list(images.keys()))

# specify which image(s) to memorize in a list
image_list = ['checkerboard']  # load a single image
#image_list = ['checkerboard', 'circle', 'cross']  # load all three images

# prepare input: choose a single image and add noise
input_image = 'checkerboard'
p_flip = 0.3  # what fraction of pixels to flip

if input_image not in image_list:
    print(f"Warning: Image '{input_image}' is not in the specified image_list.")

im_noise = utils.add_noise(images, input_image, p_flip)

# [Step 1] memorize the specified images
# TO DO: create a (N, N) matrix of synaptic weights T based on the images specified in `image_list` (~ 3 lines)
# T = ...

# initialize the state of the network
V = im_noise.flatten().astype(np.float64)  # input
E = []  # list for storing energy at each time step

if visualize:
    utils.display_images(images, image_list)

# setup plot for animation
fig, ax, img, energy_line = utils.setup_plot(size, L)

def update(frame, V, E):
    # [Step 2]: pick a single neuron at random, update its state (~ 2 lines)
    # i = ...
    # V[i] = ...

    # [Step 3]: calculate the energy of the entire network (~ 1 line)
    # energy = ...
    E.append(energy)

    img.set_data(V.reshape(size))  # update the network state
    energy_line.set_data(range(len(E)), E)  # update the energy line
    ax[1].relim() 
    ax[1].autoscale_view()
    ax[1].set_title(f"Energy: {energy.item()}")
    return [img, energy_line]

# create the animation
ani = FuncAnimation(fig, update, frames=L, blit=False, interval=10, fargs=(V, E))
plt.show()