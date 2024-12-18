This is an excercise to understand how Amari-Hopfield network works.

The notation is based on [Hopfield (PNAS, 1982)](https://www.pnas.org/doi/10.1073/pnas.79.8.2554).
Note that an identical model was proposed 10 years earlier [Amari (1972)](https://ieeexplore.ieee.org/document/1672070).

# Problem setup
We will simulate the following:
- a network of N = 256 neurons
- the network will be updated asynchronously
- patterns we want to memorize are 16 x 16 binary images

# Instructions
- `generate_images.py` contains a code to generate three images that needs to be memorized. Those images are already included in `data/images.npz` so no need to run this code.
- `utils.py` contain utility functions mainly for animation. You don't need to change this.
- `hopfield_exercise.py` contains several lines you need to implement.

# Tips
- Make sure to start with the simplest case of memorizing a single image.
- The energy function should decrease or remain the same. If it increases, there is a bug.
- The pattern where black and white patterns are reversed are also stable states of the network.

In my simulation, the checkerboard pattern no longer becomes a stable state when 3 patterns are memorized. I don't know if there is a bug in my code or whether this is what we should expect. Let me know if you have any thoughts!s