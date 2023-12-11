# RK algorithms for the navier strokes
import numpy as np
import matlab.engine
print("Lauching matlab...")
eng = matlab.engine.start_matlab()
print("Lauching finished!")

def compute_RHS(nu, dx, dz, y, ym, yg, Ny, dPdx, U, V, W):
    '''
    Fu = - d(uu)/dx -d(uv)/dy + 1/Re*du/dx + 1/Re*du/dy 
    Fv = - d(uv)/dx -d(vv)/dy + 1/Re*dv/dx + 1/Re*dv/dy 
    '''
    # Prepare some shifted values
    U_FIRST_LEFT_SHIFT = np.concatenate([U[1:, :, :], U[0, :, :][np.newaxis, :]], axis=0)
    U_FIRST_RIGHT_SHIFT = np.concatenate([U[-1, :, :][np.newaxis, :], U[:-1, :, :]], axis=0)
    U_LAST_LEFT_SHIFT = np.concatenate([U[:, :, 1:], U[:, :, 0][:, :, np.newaxis]], axis=-1)
    U_LAST_RIGHT_SHIFT = np.concatenate([U[:, :, -1][:, :, np.newaxis], U[:, :, :-1]], axis=-1)
    V_FIRST_LEFT_SHIFT = np.concatenate([V[1:, :, :], V[0, :, :][np.newaxis, :]], axis=0)
    V_FIRST_RIGHT_SHIFT = np.concatenate([V[-1, :, :][np.newaxis, :], V[:-1, :, :]], axis=0)
    V_LAST_LEFT_SHIFT = np.concatenate([V[:, :, 1:], V[:, :, 0][:, :, np.newaxis]], axis=-1)
    V_LAST_RIGHT_SHIFT = np.concatenate([V[:, :, -1][:, :, np.newaxis], V[:, :, :-1]], axis=-1)
    W_FIRST_LEFT_SHIFT = np.concatenate([W[1:, :, :], W[0, :, :][np.newaxis, :]], axis=0)
    W_FIRST_RIGHT_SHIFT = np.concatenate([W[-1, :, :][np.newaxis, :], W[0:-1, :, :]], axis=0)
    W_LAST_LEFT_SHIFT = np.concatenate([W[:, :, 1:], W[:, :, 0][:, :, np.newaxis]], axis=-1)
    W_LAST_RIGHT_SHIFT = np.concatenate([W[:, :, -1][:, :, np.newaxis], W[:, :, :-1]], axis=-1)
    
    # Compute Fu
    Fu = np.zeros_like(U)
    # Compute -d(uu)/dx
    UU = (0.5 * (U + np.roll(U, -1, axis=0)))**2
    UU_SHIFT = np.concatenate([UU[-1, :, :][np.newaxis, :], UU[0:-1, :, :]], axis=0)
    Fu -= (UU - UU_SHIFT) / dx
    # Compute -d(uv)/dy
    UV = (0.5 * (V + V_FIRST_RIGHT_SHIFT)) * (0.5 * (U[:, :-1, :] + U[:, 1:, :]))
    for i in range(1, Ny):
        Fu[:, i, :] -= (UV[:, i, :] - UV[:, i-1, :]) / (y[i] - y[i-1])
    # Compute -d(uw)/dz
    UW = (0.5 * (W + W_FIRST_RIGHT_SHIFT)) * (0.5 * (U + U_LAST_RIGHT_SHIFT))
    UW_SHIFT_LAST = np.concatenate([UW[:, :, 1:], UW[:, :, 0][:, :, np.newaxis]], axis=-1)
    Fu -= (UW_SHIFT_LAST - UW) / dz
    # Compute 1/Re*d^2u/dx^2
    Fu += nu * (U_FIRST_LEFT_SHIFT - 2*U + U_FIRST_RIGHT_SHIFT) / dx**2
    # Compute 1/Re*d^2u/dz^2
    for i in range(1, Ny):
        Fu[:, i, :] += nu * ((U[:, i+1, :] - U[:, i, :]) / (yg[i+1] - yg[i]) -
                             (U[:, i, :] - U[:, i-1, :]) / (yg[i] - yg[i-1])) / (y[i] - y[i-1])
    Fu += nu * (U_LAST_LEFT_SHIFT - 2*U + U_LAST_RIGHT_SHIFT) / dz**2
    # Add pressure gradient
    Fu += dPdx
    # Boundary conditions
    Fu[:, 0, :] = -Fu[:, 1, :]
    Fu[:, -1, :] = -Fu[:, -2, :]

    # Compute Fv
    Fv = np.zeros_like(V)
    # Compute -d(uv)/dx
    UV = (0.5 * (V + V_FIRST_RIGHT_SHIFT)) * (0.5 * (U[:, :-1, :] + U[:, 1:, :]))
    UV_FIRST_RIGHT_SHIFT = np.concatenate([UV[-1, :, :][np.newaxis, :], UV[:-1, :, :]], axis=0)
    Fv -= (UV - UV_FIRST_RIGHT_SHIFT) / dx
    
    # Compute -d(vv)/dy
    VV = (0.5 * (V[:, :-1, :] + V[:, 1:, :]))**2
    for i in range(1, Ny-1):
        Fv[:, i, :] -= (VV[:, i, :] - VV[:, i-1, :]) / (ym[i] - ym[i-1])
    
    # Compute -d(vw)/dz
    VW = 0.5 * (V + V_LAST_RIGHT_SHIFT) * 0.5 * (W[:, :-1, :] + W[:, 1:, :])
    VW_LAST_RIGHT_SHIFT = np.concatenate([VW[:, :, -1][:, :, np.newaxis], VW[:, :, :-1]], axis=-1)
    Fv -= (VW - VW_LAST_RIGHT_SHIFT) / dz
    # Compute nu*d^2v/dx^2
    Fv += nu * (V_FIRST_LEFT_SHIFT - 2*V + V_FIRST_RIGHT_SHIFT) / dx**2
    for i in range(1, Ny-1):
        Fv[:, i, :] += nu * ((V[:, i+1, :] - V[:, i, :]) / (y[i+1] - y[i]) -
                             (V[:, i, :] - V[:, i-1, :]) / (y[i] - y[i-1])) / (ym[i] - ym[i-1])
    Fv += nu * (V_LAST_LEFT_SHIFT - 2*V + V_LAST_RIGHT_SHIFT) / dz**2
    # Boundary conditions
    Fv[:, 0, :] = 0
    Fv[:, -1, :] = 0
    
    # Compute Fw
    Fw = np.zeros_like(W)
    # Compute -d(uw)/dx
    UW = 0.5 * (W + W_FIRST_RIGHT_SHIFT) * (0.5 * (U + U_LAST_RIGHT_SHIFT))
    UW_FIRST_LEFT_SHIFT =  np.concatenate([UW[1:, :, :], UW[0, :, :][np.newaxis, :]], axis=0)
    Fw -= (UW_FIRST_LEFT_SHIFT - UW) / dx
    # Compute -d(vw)/dy
    VW = (0.5 * (V + V_LAST_RIGHT_SHIFT)) * (0.5 * (W[:, :-1, :] + W[:, 1:, :]))
    for i in range(1, Ny):
        Fw[:, i, :] -= (VW[:, i, :] - VW[:, i-1, :]) / (y[i] - y[i-1])
    # Compute -d(ww)/dz
    WW = (0.5 * (W + W_LAST_LEFT_SHIFT))**2
    WW_LAST_RIGHT_SHIFT = np.concatenate([WW[:, :, -1][:, :, np.newaxis], WW[:, :, :-1]], axis=-1)
    Fw -= (WW - WW_LAST_RIGHT_SHIFT) / dz
    # Compute nu*d^2w/dx^2
    Fw += nu * (W_FIRST_LEFT_SHIFT - 2*W + W_FIRST_RIGHT_SHIFT) / dx**2
    # Compute nu*d^2w/dz^2
    for i in range(1, Ny):
        Fw[:, i, :] += nu * ((W[:, i+1, :] - W[:, i, :]) / (yg[i+1] - yg[i]) -
                             (W[:, i, :] - W[:, i-1, :]) / (yg[i] - yg[i-1])) / (y[i] - y[i-1])
    Fw = Fw + nu * ( W_LAST_LEFT_SHIFT - 2*W + W_LAST_RIGHT_SHIFT)/dz**2
    # Boundary conditions
    Fw[:, 0, :] = -Fw[:, 1, :]
    Fw[:, -1, :] = -Fw[:, -2, :]
    return Fu, Fv, Fw


def compute_pressure(Nx, Ny, Nz, dx, dz, y, kxx, kzz, DD, nu, ym, yg, dPdx, U, V, W):
    # Compute pressure term by solving Poisson equation
    # Compute the RHS of the NS equation
    RHS_u, RHS_v, RHS_w = compute_RHS(nu, dx, dz, y, ym, yg, Ny, dPdx, U, V, W)
    # RHS_p is computed at cell center
    # Compute the RHS of the pressure Poisson equation by taking the divergence
    RHS_p = np.zeros((Nx, Ny-1, Nz))
    for i in range(Ny-1):
        RHS_u_FIRST_LEFT_SHIFT = np.concatenate([RHS_u[1:, i+1, :], RHS_u[0, i+1, :][np.newaxis, :]], axis=0)
        RHS_w_LAST_LEFT_SHIFT = np.concatenate([RHS_w[:, i+1, 1:], RHS_w[:, :, 0][:, i+1, np.newaxis]], axis=-1)
        RHS_p[:, i, :] = (RHS_u_FIRST_LEFT_SHIFT - RHS_u[:, i+1, :]) / dx + \
                         (RHS_v[:, i+1, :] - RHS_v[:, i, :]) / (y[i+1] - y[i]) + \
                         (RHS_w_LAST_LEFT_SHIFT - RHS_w[:, i+1, :]) / dz

    transpose_rhsp = np.transpose(RHS_p, (0, 2, 1))
    # RHS_p_hat = np.fft.fft2(transpose_rhsp)
    RHS_p_hat = np.array(eng.fft2(np.ascontiguousarray(transpose_rhsp)))
    for i in range(Nx):
        for j in range(Nz):
            kk = kxx[i] + kzz[j]
            D = DD + np.eye(Ny-1) * kk
            if i == 0 and j == 0:
                D[0, 0] = 1.5 * D[0, 0]
            RHS_p_hat[i, j, :] = np.linalg.solve(D, np.squeeze(RHS_p_hat[i, j, :]))
    # Transform back to physical space to get pressure
    ifft_RHS_p = np.array(eng.ifft2(np.ascontiguousarray(RHS_p_hat)))
    P = np.real(np.transpose(ifft_RHS_p, (0, 2, 1)))
    return P


def matlab_fft(matrix, dim, ifft=False):
    func_name = 'fft' if not ifft else 'ifft'
    arg1 = matrix
    arg2 = []
    arg3 = dim
    eng.workspace['arg1'] = arg1
    eng.workspace['arg2'] = arg2
    eng.workspace['arg3'] = arg3
    result = eng.eval(f'{func_name}(arg1, arg2, arg3)', nargout=1)
    return result

    
def compute_projection_step(dx, dz, ym, y, kxx, kzz, Nx, Ny, Nz, DD, Uin, Vin, Win):
    # Compute divergence of velocity field
    p = np.zeros((Nx, Ny-1, Nz))
    for i in range(Ny-1):
        Uin_FIRST_LEFT_SHIFT = np.concatenate([Uin[1:, i+1, :], Uin[0, i+1, :][np.newaxis, :]], axis=0)
        Win_LAST_LEFT_SHIFT = np.concatenate([Win[:, i+1, 1:], Win[:, :, 0][:, i+1, np.newaxis]], axis=-1)
        p[:, i, :] = (Uin_FIRST_LEFT_SHIFT - Uin[:, i+1, :]) / dx + \
                    (Vin[:, i+1, :] - Vin[:, i, :]) / (y[i+1] - y[i]) + \
                    (Win_LAST_LEFT_SHIFT - Win[:, i+1, :]) / dz
    # Solve Poisson equation for p
    # Fourier transform
    fft_p = matlab_fft(np.ascontiguousarray(p), 3)
    rhs_p_hat = matlab_fft(fft_p, 1)
    rhs_p_hat = np.array(rhs_p_hat)
    # rhs_p_hat = np.fft.fft(np.fft.fft(p, axis=2), axis=0)

    # Solve Poisson equation in Fourier space
    for i in range(Nx):
        for j in range(Nz):
            kk = kxx[i] + kzz[j]
            D = DD + np.eye(Ny-1) * kk
            if i == 0 and j == 0:
                D[0, 0] = 1.5 * D[0, 0]
            rhs_p_hat[i, :, j] = np.linalg.solve(D, np.conj(rhs_p_hat[i, :, j]))

    # Inverse transform
    ifft_p = matlab_fft(np.ascontiguousarray(rhs_p_hat), 1, ifft=True)
    p = np.real(np.array(matlab_fft(ifft_p, 3, ifft=True)))
    # p = np.real(np.fft.ifft(np.fft.ifft(rhs_p_hat, axis=0), axis=2))
    # Apply fractional step
    Uout = Uin.copy()
    Vout = Vin.copy()
    Wout = Win.copy()
    p_FIRST_RIGHT_SHIFT = np.concatenate([p[-1, :, :][np.newaxis, :], p[:-1, :, :]], axis=0)
    p_LAST_RIGHT_SHIFT = np.concatenate([p[:, :, -1][:, :, np.newaxis], p[:, :, :-1]], axis=-1)
    Uout[:, 1:-1, :] = Uout[:, 1:-1, :] - (p - p_FIRST_RIGHT_SHIFT) / dx
    for i in range(1, Ny-1):
        Vout[:, i, :] = Vout[:, i, :] - (p[:, i, :] - p[:, i-1, :]) / (ym[i] - ym[i-1])
    Wout[:, 1:-1, :] = Wout[:, 1:-1, :] - (p - p_LAST_RIGHT_SHIFT) / dz

    return Uout, Vout, Wout


def apply_boundary_condition(U, V, W, Vw1, Vw2):
    U[:, 0, :] = -U[:, 1, :]
    U[:, -1, :] = -U[:, -2, :]
    V[:, 0, :] = Vw1
    V[:, -1, :] = Vw2
    W[:, 0, :] = -W[:, 1, :]
    W[:, -1, :] = -W[:, -2, :]
    return U, V, W


def compute_opposition(Pw):
    np.random.seed(0)  # Set the random seed for reproducibility
    # Vow = 10000 * np.ones(Pw.shape)
    Vow = 1. * np.random.rand(*Pw.shape)
    # Vow = 0.01 * np.random.rand(*Pw.shape)
    # TODO: Implement the remaining code if necessary
    return Vow


def time_advance_RK3(Nx, Ny, Nz, dx, dz, y, kxx, kzz, DD, nu, ym, yg, dPdx, dt, U0, V0, W0):
    # Time advance U, V, and W using RK3
    U = U0.copy()
    V = V0.copy()
    W = W0.copy()

    P = compute_pressure(Nx, Ny, Nz, dx, dz, y, kxx, kzz, DD, nu, ym, yg, dPdx, U, V, W)
    opV1 = compute_opposition(np.squeeze(0.5 * (P[:, 0, :] + P[:, 1, :])))  # opposition for bottom wall
    opV2 = compute_opposition(np.squeeze(-0.5 * (P[:, -1, :] + P[:, -2, :])))  # opposition for top wall
    # Need to remove mean to make sure no mass is lost
    opV1 = opV1 - np.mean(np.mean(opV1))
    opV2 = opV2 - np.mean(np.mean(opV2))

    # Boundary conditions
    U0, V0, W0 = apply_boundary_condition(U0, V0, W0, -0 * opV1, opV2)
    # 1st RK step
    Fu1, Fv1, Fw1 = compute_RHS(nu, dx, dz, y, ym, yg, Ny, dPdx, U, V, W)
    U = U0 + dt * 1/3 * Fu1
    V = V0 + dt * 1/3 * Fv1
    W = W0 + dt * 1/3 * Fw1

    # Projection step to correct for incompressibility
    U, V, W = compute_projection_step(dx, dz, ym, y, kxx, kzz, Nx, Ny, Nz, DD, U, V, W)

    # 2nd RK step
    Fu2, Fv2, Fw2 = compute_RHS(nu, dx, dz, y, ym, yg, Ny, dPdx, U, V, W)
    U = U0 + dt * (-Fu1 + 2 * Fu2)
    V = V0 + dt * (-Fv1 + 2 * Fv2)
    W = W0 + dt * (-Fw1 + 2 * Fw2)

    # Projection step to correct for incompressibility
    U, V, W = compute_projection_step(dx, dz, ym, y, kxx, kzz, Nx, Ny, Nz, DD, U, V, W)

    # 3rd RK step
    Fu3, Fv3, Fw3 = compute_RHS(nu, dx, dz, y, ym, yg, Ny, dPdx, U, V, W)
    U = U0 + dt * (3/4 * Fu2 + 1/4 * Fu3)
    V = V0 + dt * (3/4 * Fv2 + 1/4 * Fv3)
    W = W0 + dt * (3/4 * Fw2 + 1/4 * Fw3)

    # Projection step to correct for incompressibility
    U, V, W = compute_projection_step(dx, dz, ym, y, kxx, kzz, Nx, Ny, Nz, DD, U, V, W)

    return U, V, W
