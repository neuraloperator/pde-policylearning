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
    # plt.savefig('debug.png', dpi=800, bbox_inches='tight')  # Specify the filename and DPI
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

