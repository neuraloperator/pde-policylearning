import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio


def matrix2image(matrix, height=300, width=300, extend_value=0.5):
    matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    plt.imshow(np.squeeze(matrix), cmap='jet', interpolation='nearest', vmin=matrix.min()-extend_value, vmax=matrix.max() + extend_value)
    plt.colorbar()
    plt.tight_layout()  # Adjust the layout to prevent overlap
    # Save the heatmap image to disk
    plt.draw()
    image = np.array(plt.gcf().canvas.renderer._renderer)
    plt.close()
    # imageio.imwrite('debug.png', image)
    return image


def save_images_to_video(images, output_path, fps=15):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Specify the codec (e.g., "mp4v", "x264", "XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for image in images:
        writer.write(image[:, :, :3])
    writer.release()


def norm(matrix):
    matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    return matrix


def visualize_pressure_speed(pressure, speed_x, speed_y, extend_p=0.2, quiver_sample=2, quiver_scale=0.35, vis_img=False):
    pressure = norm(pressure)
    speed_min = min(speed_x.min(), speed_y.min())
    speed_max = max(speed_x.max(), speed_y.max())
    speed_x = (speed_x - speed_min) / (speed_max - speed_min)
    speed_y = (speed_y - speed_min) / (speed_max - speed_min)
    DOMAIN_SIZE = 1.0
    shape_y, shape_x = speed_x.shape[0], speed_x.shape[1]
    x = np.linspace(0.0, shape_y, shape_x)
    y = np.linspace(0.0, shape_x, shape_y)
    X, Y = np.meshgrid(x, y)

    plt.style.use("dark_background")
    fig_scale = (10, 6) if shape_x != shape_y else (7, 6)
    plt.figure(figsize=fig_scale)
    plt.contourf(X, Y, pressure, cmap="coolwarm", vmin=pressure.min() - extend_p, vmax=pressure.max() + extend_p)
    plt.colorbar()
    widths = np.linspace(0, 10, X.size)
    plt.quiver(X[::quiver_sample, ::quiver_sample], Y[::quiver_sample, ::quiver_sample] \
        , speed_x[::quiver_sample, ::quiver_sample], speed_y[::quiver_sample, ::quiver_sample] \
            , color="black", linewidths=widths, scale=quiver_scale, scale_units="x") 
    # scale larger, quiver smaller
    plt.tight_layout()
    plt.draw()
    image = np.array(plt.gcf().canvas.renderer._renderer)
    plt.close()
    if vis_img:
        imageio.imwrite('debug.png', image)
    return image
