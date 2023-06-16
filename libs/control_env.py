import matlab.engine
from libs.utilities3 import *
from sklearn.metrics import mean_squared_error


def to_m(numpy_a):
    if type(numpy_a) == float:
        return matlab.double(numpy_a)
    elif type(numpy_a) == int:
        return matlab.int64(numpy_a)
    else:
        return matlab.double(numpy_a.tolist())
    

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
        init_opV2 = self.rand_control(init_p)

    def add_random_noise(self, noise_scale, overwrite=False):
        if overwrite:
            self.U = np.random.normal(scale=noise_scale, size=self.U.shape)
            # self.V = np.random.normal(scale=noise_scale, size=self.V.shape)
            # self.W = np.random.normal(scale=noise_scale, size=self.W.shape)           
        else:
            self.U += np.random.normal(scale=noise_scale, size=self.U.shape)
            # self.V += np.random.normal(scale=noise_scale, size=self.V.shape)
            # self.W += np.random.normal(scale=noise_scale, size=self.W.shape)
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

    def reward_div(self, bound=-1):
        reward = - abs(np.sum(self.compute_div()))
        if reward < bound:
            reward = bound
        return reward

    def reward_gt(self, bound=-1):
        reward = 0
        reward -= mean_squared_error(self.U_gt.flatten(), self.U.flatten())
        reward -= mean_squared_error(self.V_gt.flatten(), self.V.flatten())
        reward -= mean_squared_error(self.W_gt.flatten(), self.W.flatten())
        if reward < bound:
            reward = bound
        return reward

    def reward_td(self, prev_U, prev_V, prev_W, bound=-1):
        reward = 0
        reward -= mean_squared_error(prev_U.flatten(), self.U.flatten())
        reward -= mean_squared_error(prev_V.flatten(), self.V.flatten())
        reward -= mean_squared_error(prev_W.flatten(), self.W.flatten())
        if reward < bound:
            reward = bound
        return reward

    def compute_pressure(self,):
        # this is the observation function
        P  = self.eng.compute_pressure(to_m(self.U), to_m(self.V), to_m(self.W), to_m(self.nu), to_m(self.dPdx), to_m(self.y), to_m(self.ym), \
        to_m(self.yg), to_m(self.dx), to_m(self.dz), to_m(self.kxx), to_m(self.kzz), to_m(self.Nx), to_m(self.Ny), to_m(self.Nz), to_m(self.DD))
        return np.array(P)

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
        next_state = self.compute_pressure()                    # Next state after taking the action
        div = self.reward_div()
        gt_diff = self.reward_gt()
        speed_diff = self.reward_td(prev_U, prev_V, prev_W)
        done = False                                            # Termination flag indicating if the episode is done
        info = {'-|divergence|': div, '-|now - unnoised|': gt_diff, '-|now - prev|': speed_diff} # Additional information
        return next_state, div, done, info