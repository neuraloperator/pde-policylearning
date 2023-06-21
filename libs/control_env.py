import matlab.engine
from libs.utilities3 import *
from libs.visualization import *
from sklearn.metrics import mean_squared_error


def to_m(numpy_a):
    if type(numpy_a) == float:
        return matlab.double(numpy_a)
    elif type(numpy_a) == int:
        return matlab.int64(numpy_a)
    else:
        return matlab.double(numpy_a.tolist())


def relative_loss(A, B):
    """
    Compute the relative loss between matrices A and B.
    """
    numerator = np.linalg.norm(A - B)  # Numerator: norm of the difference between A and B
    denominator = np.linalg.norm(A)  # Denominator: norm of matrix A
    
    # Handle division by zero case
    if denominator == 0:
        return 0.0
    
    # Compute relative loss
    loss = numerator / denominator
    
    return loss


class NSControl:
    def __init__(self, timestep, noise_scale):
        print("Lauching matlab...")
        self.eng = matlab.engine.start_matlab()
        print("Lauching finished!")
        self.eng.addpath("./libs/matlab_codes")
        
        # Initialize with saved field
        init_minchan_path = './data/channel180_minchan.mat'

        # Load the .mat file
        mat_data = scipy.io.loadmat(init_minchan_path)
       
        # Access the fields
        self.x = mat_data['x']
        self.y = mat_data['y']
        self.z = mat_data['z']
        self.xm = mat_data['xm']
        self.ym = mat_data['ym']
        self.zm = mat_data['zm']
        self.UU = mat_data['UU']
        self.VV = mat_data['VV']
        self.WW = mat_data['WW']  # BS, W, H

        # Global variables
        self.nu = 1 / 0.32500000E+04  # kinematic viscosity
        self.dPdx = 0.57231059E-01**2  # pressure gradient (utau^2)
        self.Nt = timestep  # number of time steps
        self.dt = 0.001  # time step
        self.dx = self.x[1] - self.x[0]
        self.dz = self.z[1] - self.z[0]
        self.yg = np.concatenate(([-self.ym[0]], self.ym, [2 + self.ym[0]]))
        self.Nx = len(self.x) - 2
        self.Nz = len(self.z) - 2
        self.Ny = len(self.y)

        self.U = self.UU[0:self.Nx, :, 1:self.Nz+1]
        self.V = self.VV[1:self.Nx+1, :, 1:self.Nz+1]
        self.W = self.WW[1:self.Nx+1, :, 0:self.Nz]
        self.U_gt = self.U.copy()
        self.V_gt = self.V.copy()
        self.W_gt = self.W.copy()
        np.random.seed(0)
        self.s = np.random.default_rng()
        # Call matlab main directly.
        # status = self.eng.main(to_m(x),to_m(y),to_m(z),to_m(xm),to_m(ym),to_m(zm),to_m(UU),to_m(VV),to_m(WW))

        # Define modified wavenumbers
        self.kxx = np.zeros(self.Nx)
        self.kzz = np.zeros(self.Nz)
        for k in range(int(self.Nx/2)+1):
            self.kxx[k] = 2 * (np.cos(2 * np.pi * k / self.Nx) - 1) / self.dx**2
        for k in range(int(self.Nx/2)+1, self.Nx):
            self.kxx[k] = 2 * (np.cos(2 * np.pi * (-self.Nx + k) / self.Nx) - 1) / self.dx**2

        for k in range(int(self.Nz/2)+1):
            self.kzz[k] = 2 * (np.cos(2 * np.pi * k / self.Nz) - 1) / self.dz**2
        for k in range(int(self.Nz/2)+1, self.Nz):
            self.kzz[k] = 2 * (np.cos(2 * np.pi * (-self.Nz + k) / self.Nz) - 1) / self.dz**2

        # Define matrix for y direction Poisson solver
        self.DD = np.zeros((self.Ny-1, self.Ny-1))
        for j in range(self.Ny-1):
            self.DD[j, j] = -1 / (self.y[j+1] - self.y[j]) * (1 / (self.yg[j+2] - self.yg[j+1]) +
                                                        1 / (self.yg[j+1] - self.yg[j]))

        for j in range(self.Ny-2):
            self.DD[j+1, j] = 1 / (self.y[j+2] - self.y[j+1]) / (self.yg[j+2] - self.yg[j+1])
            self.DD[j, j+1] = 1 / (self.y[j+1] - self.y[j]) / (self.yg[j+2] - self.yg[j+1])

        self.DD[0, 0] += 1 / (self.y[1] - self.y[0]) / (self.yg[1] - self.yg[0])
        self.DD[-1, -1] += 1 / (self.y[self.Ny-1] - self.y[self.Ny-2]) / (self.yg[self.Ny] - self.yg[self.Ny-1])

        self.add_random_noise(noise_scale)
        print(f"Initially, the divergence is {self.reward_div()}, the mse is {self.reward_gt()}")
        init_p = self.compute_pressure()
        # Used in normalization
        self.speed_min = min(self.U.min(), self.V.min(), self.W.min())
        self.speed_max = max(self.U.max(), self.V.max(), self.W.max())
        self.p_min = max(-2.0, init_p.min())
        self.p_max = min(init_p.max(), 1.5)
        # init_opV2 = self.rand_control(init_p)

    def add_random_noise(self, noise_scale, overwrite=False):
        if overwrite:
            self.U = np.random.normal(scale=noise_scale, size=self.U.shape)
            self.V = np.random.normal(scale=noise_scale, size=self.V.shape)
            self.W = np.random.normal(scale=noise_scale, size=self.W.shape)           
        else:
            self.U += np.random.normal(scale=noise_scale, size=self.U.shape)
            self.V += np.random.normal(scale=noise_scale, size=self.V.shape)
            self.W += np.random.normal(scale=noise_scale, size=self.W.shape)
        return

    def compute_div(self):
        div = np.zeros((self.Nx, self.Ny-1, self.Nz))
        uxsum, uysum, uzsum = 0, 0, 0
        for j in range(self.Ny-1):
            prev_u, prev_w = self.U[:, j+1, :], self.W[:, j+1, :]
            next_u, next_w = np.concatenate([self.U[1:, j+1, :], self.U[0, j+1, :][np.newaxis, :]], axis=0), \
                            np.concatenate([self.W[:, j+1, 1:], self.W[:, j+1, 0][:, np.newaxis]], axis=1)
            ux = (next_u - prev_u) / self.dx 
            uy = (self.V[:, j+1, :] - self.V[:, j, :]) / (self.y[j+1] - self.y[j])
            uz = (next_w - prev_w) / self.dz
            # print(f"uxyz is: {np.sum(abs(ux))}, {np.sum(uy)}, {np.sum(uz)}. ")
            uxsum += np.abs(ux)
            uysum += np.abs(uy)
            uzsum += np.abs(uz)
            div[:, j, :] = ux + uy + uz
        return div

    def reward_div(self, bound=-100):
        reward = - abs(np.sum(self.compute_div()))
        if reward < bound:
            reward = bound
        return reward

    def reward_gt(self, bound=-100):
        reward = 0
        reward -= relative_loss(self.U_gt.flatten(), self.U.flatten())
        reward -= relative_loss(self.V_gt.flatten(), self.V.flatten())
        reward -= relative_loss(self.W_gt.flatten(), self.W.flatten())
        if reward < bound:
            reward = bound
        return reward

    def reward_td(self, prev_U, prev_V, prev_W, bound=-100):
        reward = 0
        reward -= relative_loss(prev_U.flatten(), self.U.flatten())
        reward -= relative_loss(prev_V.flatten(), self.V.flatten())
        reward -= relative_loss(prev_W.flatten(), self.W.flatten())
        if reward < bound:
            reward = bound
        return reward

    def compute_pressure(self,):
        # this is the observation function
        self.P  = self.eng.compute_pressure(to_m(self.U), to_m(self.V), to_m(self.W), to_m(self.nu), to_m(self.dPdx), to_m(self.y), to_m(self.ym), \
        to_m(self.yg), to_m(self.dx), to_m(self.dz), to_m(self.kxx), to_m(self.kzz), to_m(self.Nx), to_m(self.Ny), to_m(self.Nz), to_m(self.DD))
        self.P = np.array(self.P)
        return self.P

    def get_state(self):
        pressure = self.compute_pressure()                    # Next state after taking the action
        next_state = np.squeeze(-0.5 * (pressure[:, -1, :] + pressure[:, -2, :]))
        return next_state
    
    def vis_state(self, vis_img=False, sample_slice_top=15, sample_slice_others=10):
        p_min, p_max = -0.05, 0.05
        pressure = self.compute_pressure()
        cut_dim = self.U.shape[0]
        # get front view
        mid_index = pressure.shape[2] // sample_slice_others
        front_pressure = pressure[:, :cut_dim, mid_index].transpose()
        u_in_xy = self.U[:, :cut_dim, mid_index].transpose()
        v_in_xy = self.V[:, :cut_dim, mid_index].transpose()
        front_view = visualize_pressure_speed(front_pressure, pressure_min=p_min, pressure_max=p_max, \
            speed_horizontal=u_in_xy, speed_vertical=v_in_xy, vis_img=vis_img, vis_name='front', quiver_scale=0.3, \
            x_sample_interval=2, y_sample_interval=2)
        
        # get top view
        mid_index = pressure.shape[1] // sample_slice_top
        top_pressure = np.squeeze(-0.5 * (pressure[:, -1, :] + pressure[:, -2, :]))
        u_in_xz = self.U[:, mid_index, :]
        w_in_xz = self.W[:, mid_index, :]
        top_view = visualize_pressure_speed(top_pressure, pressure_min=p_min, pressure_max=p_max, \
            speed_horizontal=u_in_xz, speed_vertical=w_in_xz, vis_img=vis_img, quiver_scale=0.06, vis_name='top',)

        # get side view
        sample_index = pressure.shape[0] // sample_slice_others
        side_pressure = pressure[sample_index, :cut_dim, :]
        v_in_yz = self.V[sample_index, :cut_dim, :]
        w_in_yz = self.W[sample_index, :cut_dim, :]
        side_view = visualize_pressure_speed(side_pressure, pressure_min=p_min, pressure_max=p_max, \
            speed_horizontal=w_in_yz, speed_vertical=v_in_yz, vis_img=vis_img, vis_name='side', \
                quiver_scale=0.3, x_sample_interval=2, y_sample_interval=2)
        if vis_img:
            import pdb; pdb.set_trace()
        return top_view, front_view, side_view
    
    def rand_control(self, P):
        opV2 = self.eng.compute_opposition(to_m(P))
        return np.array(opV2)

    def step(self, opV2):
        # Perform one step in the environment
        # Update state, calculate reward, check termination condition, etc.
        # Return the next state, reward, termination flag, and additional info
        prev_U, prev_V, prev_W = self.U.copy(), self.V.copy(), self.W.copy()
        U, V, W = self.eng.time_advance_RK3(opV2, to_m(self.U), to_m(self.V), to_m(self.W), to_m(self.nu), to_m(self.dPdx), \
        to_m(self.y), to_m(self.ym), to_m(self.yg), to_m(self.dx), to_m(self.dz), to_m(self.dt), to_m(self.kxx), to_m(self.kzz), \
            to_m(self.Nx), to_m(self.Ny), to_m(self.Nz), to_m(self.DD), nargout=3)
        self.U, self.V, self.W = np.array(U), np.array(V), np.array(W)
        next_state = self.get_state()
        div = self.reward_div()
        gt_diff = self.reward_gt()
        speed_diff = self.reward_td(prev_U, prev_V, prev_W)
        done = False                                                                                # Termination flag indicating if the episode is done
        info = {'-|divergence|': div, '-|now - unnoised| / ｜now|': gt_diff, '-|now - prev| / |now|': speed_diff}    # Additional information
        return next_state, div, done, info