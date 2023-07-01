import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import wandb


def matrix2image(matrix, height=300, width=300, extend_value=0.5, eps=1e-9, normalize=False):
    if normalize:
        if matrix.max() - matrix.min() < eps:
            matrix = matrix
        else:
            matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    plt.imshow(np.squeeze(matrix), cmap='jet', interpolation='nearest', vmin=-extend_value, vmax=extend_value)
    plt.colorbar()
    plt.tight_layout()  # Adjust the layout to prevent overlap
    # Save the heatmap image to disk
    plt.draw()
    image = np.array(plt.gcf().canvas.renderer._renderer)
    plt.close()
    # imageio.imwrite('outputs/debug.png', image)
    # import pdb; pdb.set_trace()
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


def visualize_pressure_speed(pressure, pressure_min, pressure_max, speed_horizontal, speed_vertical, \
                             extend_p=0.2, quiver_interval=2, quiver_scale=0.35, vis_img=False, vis_name='top', \
                                 x_sample_interval=2, y_sample_interval=2):
    pressure[pressure > pressure_max] = pressure_max
    pressure[pressure < pressure_min] = pressure_min
    shape_y, shape_x = speed_horizontal.shape[0], speed_horizontal.shape[1]
    x = np.linspace(0.0, shape_y, shape_x)
    y = np.linspace(0.0, shape_x, shape_y)
    X, Y = np.meshgrid(x, y)
    y_sample_index = list(range(1, shape_y, y_sample_interval))
    x_sample_index = list(range(1, shape_x, x_sample_interval))
    plt.style.use("dark_background")
    fig_scale = (10, 6) if shape_x != shape_y else (7, 6)
    plt.figure(figsize=fig_scale)
    v = np.linspace(pressure_min, pressure_max, 10, endpoint=True)
    plt.contourf(X, Y, pressure, v, cmap="coolwarm")
    colorbar = plt.colorbar(ticks=np.arange(pressure_min, pressure_max, (pressure_max - pressure_min) / 10))
    plt.clim(pressure_min, pressure_max)
    widths = np.linspace(0, 10, X.size)
    plt.quiver(X[y_sample_index, :][:, x_sample_index], Y[y_sample_index, :][:, x_sample_index] \
        , speed_horizontal[y_sample_index, :][:, x_sample_index], speed_vertical[y_sample_index, :][:, x_sample_index] \
            , color="black", linewidths=widths, scale=quiver_scale, scale_units="x") # scale larger, quiver smaller
    plt.tight_layout()
    plt.draw()
    image = np.array(plt.gcf().canvas.renderer._renderer)
    plt.close()
    if vis_img:
        imageio.imwrite(f'outputs/{vis_name}.png', image)
    image = image[:, :, [2, 1, 0, 3]]  # switch channel
    return image


def vis_diagram(dat):
    data_num = dat['y'].shape[0]
    for index in range(0, data_num - 1, 5):
        vmin = dat['y'][index, :, :].min()
        vmax = dat['y'][index, :, :].max()
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 4))
        plt.subplot(1, 3, 1)
        im1 = plt.imshow(dat['x'][index, :, :, 0], cmap='jet', aspect='auto')
        plt.title('Input')
        plt.subplot(1, 3, 2)
        im2 = plt.imshow(dat['y'][index, :, :], cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
        plt.title('True Output')
        plt.subplot(1, 3, 3)
        im3 = plt.imshow(dat['pred'][index, :, :], cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
        plt.title('Prediction')
        cbar_ax = fig.add_axes([.92, 0.15, 0.04, 0.7])
        fig.colorbar(im3, cax=cbar_ax)
        wandb.log({f"data_id_{index}": plt})
    return