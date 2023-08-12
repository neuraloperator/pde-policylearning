import wandb
import matlab.engine
from libs.utilities3 import *
from libs.visualization import *
from sklearn.metrics import mean_squared_error
from libs.env_util import to_m, relative_loss, apply_periodic_boundary


class NSControlEnvMatlab:
    def __init__(self, args):
        self.args = args
        self.Re = args.Re
        self.nu = 1 / 0.32500000E+04
        self.default_re = 178.1899
        if self.Re is not None:
            self.nu = self.nu * (self.default_re / self.Re)
        self.control_timestep = args.control_timestep
        self.detect_plane = args.detect_plane
        self.test_plane = args.test_plane
        self.w_weight = args.w_weight
        self.bc_type = args.bc_type
        print("Lauching matlab...")
        self.eng = matlab.engine.start_matlab()
        print("Lauching finished!")
        self.eng.addpath("./libs/matlab_codes")
        self.load_state(load_path=args.init_cond_path)
        save_path = './outputs/stable_flow.npy'
        self.dump_state(save_path=save_path)
        self.load_state(load_path=save_path)
        self.v_scale, self.w_scale = 10, 10
        self.V = self.V
        self.W = self.W
        self.U_gt = self.U.copy()
        self.V_gt = self.V.copy()
        self.W_gt = self.W.copy()
        np.random.seed(0)
        self.s = np.random.default_rng()

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
        
        '''
        Calculate initialized mean bulk velocity
        '''
        self.meanU0 = self.cal_bulk_v()
        print(f"Initially, the divergence is {self.reward_div()}.")
        init_p = self.cal_pressure()
        # Used in normalization
        self.speed_min = min(self.U.min(), self.V.min(), self.W.min())
        self.speed_max = max(self.U.max(), self.V.max(), self.W.max())
        self.p_min = max(-2.0, init_p.min())
        self.p_max = min(init_p.max(), 1.5)
        self.info_init = None
        
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
    
    '''
    Save and load state.
    '''
    
    def dump_state(self, save_path):
        mat_data = {
            'x': np.array(self.x),
            'y': np.array(self.y),
            'z': np.array(self.z),
            'xm': np.array(self.xm),
            'ym': np.array(self.ym),
            'zm': np.array(self.zm),
            'U': np.array(self.U),
            'V': np.array(self.V),
            'W': np.array(self.W),
        }
        scipy.io.savemat(save_path, mat_data)
        return
    
    def load_state(self, load_path='./data/channel180_minchan.mat'):
        # Load the .mat file
        mat_data = scipy.io.loadmat(load_path)
        # Access the fields
        self.x = mat_data['x']
        self.y = mat_data['y']
        self.z = mat_data['z']
        self.xm = mat_data['xm']
        self.ym = mat_data['ym']
        self.zm = mat_data['zm']
        
        # Global variables
        self.dPdx = 0.57231059E-01**2  # pressure gradient (utau^2)
        self.dt = 0.001  # time step
        self.dx = self.x[1] - self.x[0]
        self.dz = self.z[1] - self.z[0]
        self.yg = np.concatenate(([-self.ym[0]], self.ym, [2 + self.ym[0]]))
        self.Nx = len(self.x) - 2
        self.Nz = len(self.z) - 2
        self.Ny = len(self.y)

        if 'UU' in mat_data:
            UU = mat_data['UU']
            VV = mat_data['VV']
            WW = mat_data['WW']  # BS, W, H
            self.U = UU[0:self.Nx, :, 1:self.Nz+1]
            self.V = VV[1:self.Nx+1, :, 1:self.Nz+1]
            self.W = WW[1:self.Nx+1, :, 0:self.Nz]
        else:
            self.U = mat_data['U']
            self.V = mat_data['V']
            self.W = mat_data['W']
        
    '''
    Calculating scores.
    '''  
    
    def cal_div(self):
        div1 = np.zeros((self.Nx, self.Ny-1, self.Nz))
        uxsum, uysum, uzsum = 0, 0, 0
        for j in range(self.Ny-1):
            prev_u, prev_w = self.U[:, j+1, :], self.W[:, j+1, :]
            next_u, next_w = np.concatenate([self.U[1:, j+1, :], self.U[0, j+1, :][np.newaxis, :]], axis=0), \
                            np.concatenate([self.W[:, j+1, 1:], self.W[:, j+1, 0][:, np.newaxis]], axis=1)
            ux = (next_u - prev_u) / self.dx 
            uy = (self.V[:, j+1, :] - self.V[:, j, :]) / (self.y[j+1] - self.y[j])
            uz = (next_w - prev_w) / self.dz
            uxsum += np.abs(ux)
            uysum += np.abs(uy)
            uzsum += np.abs(uz)
            div1[:, j, :] = ux + uy + uz
        div = self.eng.compute_div(to_m(self.U), to_m(self.V), to_m(self.W), to_m(self.dx), to_m(self.y), to_m(self.dz),\
            to_m(self.Nx), to_m(self.Ny), to_m(self.Nz))
        div = np.array(div)
        return div

    def cal_pressure(self,):
        # this is the observation function
        self.P  = self.eng.compute_pressure(to_m(self.U), to_m(self.V), to_m(self.W), to_m(self.nu), to_m(self.dPdx), to_m(self.y), to_m(self.ym), \
            to_m(self.yg), to_m(self.dx), to_m(self.dz), to_m(self.kxx), to_m(self.kzz), to_m(self.Nx), to_m(self.Ny), to_m(self.Nz), to_m(self.DD))
        self.P = np.array(self.P)
        return self.P

    def cal_dpdx_finite_difference(self, pressure_top):
        grad_total, num = 0, 0
        for select_index in range(pressure_top.shape[0] - 1):
            pressure_gradient = (pressure_top[select_index + 1, :] - pressure_top[select_index, :]) / self.dx[0]
            grad_total += abs(pressure_gradient).mean()
            num += 1
        mean_pg = abs(grad_total / num)
        return mean_pg

    def cal_bulk_v(self):
        meanU = self.eng.calculate_meanU(to_m(self.ym), to_m(self.U))
        meanU = np.array(meanU).item()
        return meanU

    def cal_velocity_mean(self, velocity_name='U', sample_index=10):
        if velocity_name == 'U':
            all_velocity = self.U
        elif velocity_name == 'V':
            all_velocity = self.V
        elif velocity_name == 'W':
            all_velocity = self.W
        else:
            raise RuntimeError("not supported velocity!")
        if sample_index is not None:
            velocity = abs(all_velocity)[:, -sample_index:, :].mean()
        else:
            velocity = abs(all_velocity).mean()
        return velocity
 
    def cal_speed_norm(self, ):
        return np.linalg.norm(self.V) + np.linalg.norm(self.U) + np.linalg.norm(self.W)
    
    def cal_dudy(self, ):
        dudy_all = []
        for select_index in range(self.U.shape[1] - 2):
            dudy = (self.U[:, select_index + 1, :] - self.U[:, select_index, :]) / \
            (self.y[select_index + 1][0] - self.y[select_index][0])
            dudy_all.append(dudy)
        return dudy_all
    
    def cal_shear_stress(self, ):
        # -u*v + nu * (dU/dy)
        wall_u = self.U[:, -1, :]
        wall_v = self.V[:, -1, :]
        dudy_all = self.cal_dudy()
        dudy = dudy_all[-1]
        first_term = -wall_u * wall_v
        second_term = self.nu * dudy
        shear_stress = first_term + second_term
        shear_stress_mean = np.mean(shear_stress)
        shear_stress_res = abs(shear_stress_mean)
        return shear_stress_res
    
    def reward_div(self, bound=-100):
        reward = - abs(np.sum(self.cal_div()))
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

    def cal_relative_info(self, info):
        if self.info_init is None:
            self.info_init = info
            return {}
        else:
            relative_dict = {}
            for one_k in info:
                new_k = one_k.replace("drag_reduction", "drag_reduction_relative")
                relative_dict[new_k] = info[one_k] / self.info_init[one_k]
            return relative_dict
    

    '''
    Visualizations
    '''
    
    def vis_state(self, vis_img=False, sample_slice_top=15, sample_slice_others=10):
        pressure = self.cal_pressure()
        cut_dim = self.U.shape[0]
        # get front view
        mid_index = pressure.shape[2] // sample_slice_others
        front_pressure = pressure[:, -cut_dim:, mid_index].transpose()
        u_in_xy = self.U[:, -cut_dim:, mid_index].transpose() / 10000
        v_in_xy = self.V[:, -cut_dim:, mid_index].transpose()
        front_view = visualize_pressure_speed(front_pressure, pressure_min=-0.01, pressure_max=0.01, \
            speed_horizontal=u_in_xy, speed_vertical=v_in_xy, vis_img=vis_img, vis_name='front', quiver_scale=0.03, \
            x_sample_interval=2, y_sample_interval=2, v_flip=False)

        # TODO: check top view and side view visualizations' order
        # get top view
        mid_index = pressure.shape[1] // sample_slice_top
        top_pressure = np.squeeze(-0.5 * (pressure[:, -1, :] + pressure[:, -2, :]))
        u_in_xz = self.U[:, -mid_index, :] - self.U[:, -mid_index, :].mean()
        w_in_xz = self.W[:, -mid_index, :]
        top_view = visualize_pressure_speed(top_pressure, pressure_min=-0.03, pressure_max=0.03, \
            speed_horizontal=u_in_xz, speed_vertical=w_in_xz, vis_img=vis_img, quiver_scale=0.05, vis_name='top',)

        # get side view
        sample_index = pressure.shape[0] // sample_slice_others
        side_pressure = pressure[sample_index, :cut_dim, :]
        v_in_yz = self.V[sample_index, :cut_dim, :]
        w_in_yz = self.W[sample_index, :cut_dim, :]
        side_view = visualize_pressure_speed(side_pressure, pressure_min=-0.005, pressure_max=0.005, \
            speed_horizontal=w_in_yz, speed_vertical=v_in_yz, vis_img=vis_img, vis_name='side', \
                quiver_scale=0.03, x_sample_interval=2, y_sample_interval=2)
        if vis_img:
            import pdb; pdb.set_trace()
        return top_view, front_view, side_view
    
    def plot_spatial_distribution(self, step_index):
        info = {}
        dudy_all = self.cal_dudy()
        for chart_key in ['U', 'V', 'W', 'dudy']:
            cur_data = []
            for sample_index in range(30):
                if chart_key in ['U', 'V', 'W']:
                    value_name = chart_key + "_velocity"
                    value = self.cal_velocity_mean(chart_key, sample_index=-sample_index)
                elif chart_key == 'dudy':
                    value_name = 'dudy'
                    value = np.mean(abs(dudy_all[-sample_index]))
                else:
                    raise RuntimeError()
                cur_data.append([sample_index, value])
                
            # Create a table with the columns to plot
            table = wandb.Table(data=cur_data, columns=["layer index", value_name])

            # Use the table to populate various custom charts
            line_plot = wandb.plot.line(table, x="layer index", y=value_name, title="y-aixs distribution of " + value_name)
            info['spatial_dist/' + str(step_index) + "/" + value_name] = line_plot
        wandb.log(info)
        return
    
    '''
    Control policies.
    '''
    
    def reset_init(self):
        self.info_init = None
    
    def rand_control(self, P):
        opV2 = self.eng.compute_opposition(to_m(P))
        opV2 = np.array(opV2)
        return opV2
    
    def gt_control(self):
        opV1 = self.V[:, self.detect_plane, :]
        opV1 = np.array(opV1)
        opV2 = self.V[:, -self.detect_plane, :]
        opV2 = np.array(opV2)
        return opV1, opV2

    def get_boundary_pressures(self):
        pressure = self.cal_pressure()                    # Next state after taking the action
        p1 = np.squeeze(-0.5 * (pressure[:, 0, :] + pressure[:, 1, :]))
        p2 = np.squeeze(-0.5 * (pressure[:, -1, :] + pressure[:, -2, :]))
        return p1, p2
    
    def step_rk3(self, opV1, opV2):
        U, V, W, dPdx = self.eng.time_advance_RK3(to_m(opV1), to_m(opV2), to_m(self.U), to_m(self.V), to_m(self.W), to_m(self.meanU0), to_m(self.nu), to_m(self.dPdx), \
        to_m(self.y), to_m(self.ym), to_m(self.yg), to_m(self.dx), to_m(self.dz), to_m(self.dt), to_m(self.kxx), to_m(self.kzz), \
            to_m(self.Nx), to_m(self.Ny), to_m(self.Nz), to_m(self.DD), nargout=4)
        self.U, self.V, self.W, self.dPdx = np.array(U), np.array(V), np.array(W), np.array(dPdx).item()
        
    def step(self, opV1, opV2):
        # Perform one step in the environment
        # Update state, calculate reward, check termination condition, etc.
        # Return the next state, reward, termination flag, and additional info
        prev_U, prev_V, prev_W = self.U.copy(), self.V.copy(), self.W.copy()
        for i in range(2):
            applied_opV1 = opV1 * 0
            self.step_rk3(applied_opV1, opV2)
        p1, p2 = self.get_boundary_pressures()
        div = self.reward_div()
        dpdx_finite_difference = self.cal_dpdx_finite_difference(p2)
        u_velocity = self.cal_bulk_v()
        v_velocity = self.cal_velocity_mean('V', sample_index=None)
        w_velocity = self.cal_velocity_mean('W', sample_index=None)
        pressure_mean = p2.mean()
        gt_diff = self.reward_gt()
        speed_diff = self.reward_td(prev_U, prev_V, prev_W)
        speed_norm = self.cal_speed_norm()
        shear_stress = self.cal_shear_stress()
        done = False
        info = {
                'drag_reduction/1_shear_stress': shear_stress,
                'drag_reduction/2_1_mass_flow': u_velocity,
                'drag_reduction/2_2_v_velocity': v_velocity,
                'drag_reduction/2_3_w_velocity': w_velocity,
                'drag_reduction/3_1_pressure_mean': pressure_mean,
                'drag_reduction/3_2_dPdx_finite_difference': dpdx_finite_difference,
                'drag_reduction/3_3_dPdx_reverse_cal': self.dPdx,
                'drag_reduction/4_1_-|divergence|': div, 
                'drag_reduction/4_2_-|now - unnoised| ÷ ｜now|': gt_diff, 
                'drag_reduction/4_3_-|now - prev| ÷ |now|': speed_diff, 
                'drag_reduction/4_4_speed_norm': speed_norm,
                }
        norm_info = self.cal_relative_info(info)
        info.update(norm_info)
        return p2, div, done, info
