import numpy as np
from mayavi import mlab
import scipy.io
import os
import cv2
# create the folder
os.makedirs("data/exp", exist_ok=True)

def load_step_data_and_cal_dis(step):
    mat_data = scipy.io.loadmat(f'data/channel180_minchan_{step}.mat')

    # Extract U, V, and W arrays
    U = mat_data['U']
    V = mat_data['V']
    W = mat_data['W']

    # Convert U, V, and W to NumPy arrays
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)

    # Filter y direction
    filter_num = 15
    U = U[:, :filter_num, :]
    V = V[:, :filter_num, :]
    W = W[:, :filter_num, :]

    # Calculate the velocity gradients
    dU_dx, dU_dy, dU_dz = np.gradient(U, axis=(0, 1, 2))
    dV_dx, dV_dy, dV_dz = np.gradient(V, axis=(0, 1, 2))
    dW_dx, dW_dy, dW_dz = np.gradient(W, axis=(0, 1, 2))

    # Assemble the velocity gradient tensor
    gradient_tensor = np.stack((dU_dx, dU_dy, dU_dz, dV_dx, dV_dy, dV_dz, dW_dx, dW_dy, dW_dz), axis=-1)
    gradient_tensor = gradient_tensor.reshape(dU_dx.shape + (3, 3))

    # Calculate the discriminant following Topology of fine-scale motions in turbulent channel flow
    trace_gradient_tensor = np.trace(gradient_tensor, axis1=-2, axis2=-1)  # This will give you a 3D array of shape (Nx, Ny, Nz)
    determinant_gradient_tensor = np.linalg.det(gradient_tensor)  # This will give you a 3D array of shape (Nx, Ny, Nz)

    reshaped_gradient_tensor = gradient_tensor.reshape(gradient_tensor.shape[:-2] + (3, 3))
    elementwise_product = np.einsum('...ij,...jk->...ik', reshaped_gradient_tensor, reshaped_gradient_tensor)
    trace_elementwise_product = np.trace(elementwise_product, axis1=-2, axis2=-1)  # This will give you a 3D array of shape (Nx, Ny, Nz)

    R_value = - trace_gradient_tensor
    Q_value = 0.5 * (trace_gradient_tensor ** 2 - trace_elementwise_product)
    discriminant = 27/4*R_value**2 + Q_value ** 3
    return discriminant

init_discriminant = load_step_data_and_cal_dis('1')
step_list = [str(i) for i in range(1, 11201, 200)]
frame_length = 30
decay_thre = 1e-4

for frame_id in range(frame_length):
    # Create an isosurface
    step_id = step_list[int(frame_id / frame_length * len(step_list))]
    discriminant = load_step_data_and_cal_dis(step_id)
    mlab.figure('vid', bgcolor=(1, 1, 1), fgcolor=(0.,0.,0.), size=(800, 600))
    mlab.contour3d(discriminant, contours=[frame_id * decay_thre + 1e-5], opacity=0.2, colormap='viridis')
    mlab.contour3d(init_discriminant, contours=[1e-5 + 1e-5], opacity=0.0, colormap='viridis')  # dummy background to hold the axes
    # mlab.contour3d(discriminant, contours=[0.0], opacity=1.0, colormap='jet')
    # mlab.contour3d(discriminant, contours=[0.0], opacity=1.0, colormap='hot')
    # Add axes
    mlab.axes(color=(0.0, 0.0, 0.0), line_width=1.5, ranges=(0, 32, 0, 16, 0, 32))
    mlab.xlabel('x')
    mlab.ylabel('y')
    mlab.zlabel('z')
    mlab.view(azimuth=-90, elevation=-150, distance=80)
    mlab.savefig('data/exp/{:03d}.png'.format(frame_id))
    # mlab.show()
    mlab.clf()


# Define the path to the images and the output video file
image_folder = 'data/exp/'
video_name = 'output_video.mp4'

# Get the list of image files
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))

# Set the frame width, height width 
# the width, height of first image
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

print(f"Video is saved at: {video_name}.")
cv2.destroyAllWindows()
video.release()