import matplotlib.pyplot as plt
import numpy as np

def setup_plot(size, L):
    """sets up the matplotlib plot for animation

    Args:
        size (tuple): Dimensions of the neural state image (e.g., (16, 16))
        L (int): Number of time steps for the animation

    Returns:
        fig (matplotlib.figure.Figure): figure object
        ax (list): list of axes for subplots
        img (matplotlib.image.AxesImage): network state image
        energy_line (matplotlib.lines.Line2D): energy plot line
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # subplot for state animation
    img = ax[0].imshow(np.zeros(size), cmap='gray', vmin=0, vmax=1)
    ax[0].axis('off')
    ax[0].set_title("Neural State")

    # subplot for energy evolution
    energy_line, = ax[1].plot([], [], lw=2)
    ax[1].set_xlim(0, L)
    ax[1].set_title("Energy")
    ax[1].set_xlabel("Time Step")

    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)

    return fig, ax, img, energy_line


def display_images(images, image_list):
    """displays a series of images with their corresponding names

    Args:
        images (dict): dictionary of images with their names as keys
        image_list (list): list of image names to display
    """
    filtered_images = {name: images[name] for name in image_list if name in images}
    
    fig, ax = plt.subplots(1, len(filtered_images), figsize=(5 * len(filtered_images), 5))
    if len(filtered_images) == 1:
        ax = [ax]  # Ensure ax is always iterable
    for idx, (name, image) in enumerate(filtered_images.items()):
        ax[idx].imshow(image, cmap='gray')
        ax[idx].set_title(f"{name}")
        ax[idx].axis('off')
    plt.show()


def add_noise(images, image_name, p_flip):
    """corrupts a specific image by flipping pixels with a given probability

    Args:
        images (dict): dictionary of images with their names as keys
        image_name (str): name of the image to corrupt
        p_flip (float): probability of flipping each pixel

    Returns:
        im_noise (np.array): corrupted image
    """
    if image_name not in images:
        raise ValueError(f"Image '{image_name}' not found in the provided images.")
    
    image = images[image_name]
    flip_mask = np.random.rand(*image.shape) < p_flip
    im_noise = np.where(flip_mask, 1 - image, image)
    return im_noise