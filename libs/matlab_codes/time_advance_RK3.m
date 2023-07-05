function [U, V, W] = time_advance_RK3(U0,V0,W0,nu,dPdx,y,ym,yg,dx,dz,dt,kxx,kzz,Nx,Ny,Nz,DD)
%% Time advance U, V and W using RK3

U  = U0;
V  = V0;
W  = W0;

% P    = compute_pressure(U,V,W,nu,dPdx,y,ym,yg,dx,dz,kxx,kzz,Nx,Ny,Nz,DD);
% opV1 = compute_opposition(squeeze( 0.5*(P(:,  1,:)+P(:,    2,:)))); % opposition for bottom wall
% opV2 = compute_opposition(squeeze(-0.5*(P(:,end,:)+P(:,end-1,:)))); % opposition for top wall

% need to remove mean to make sure no mass is lost
% opV1 = opV1 - mean(mean(opV1));

% commented out by Zelin, do not apply bc here.
%opV2 = opV2 - mean(mean(opV2));
%opV1 = opV2 * 0;
% boundary conditions
%[U0, V0, W0] = apply_boundary_condition(U0,V0,W0,-0*opV1,opV2);

% 1st RK step
[Fu1, Fv1, Fw1] = compute_RHS(U,V,W,nu,dx,dz,y,ym,yg,Ny,dPdx);

U = U0 + dt*1/3*Fu1;
V = V0 + dt*1/3*Fv1;
W = W0 + dt*1/3*Fw1;

% Projection step to correct for incompressibility
[U, V, W] = compute_projection_step(U,V,W,dx,dz,ym,y,kxx,kzz,Nx,Ny,Nz,DD);

% 2nd RK step
[Fu2, Fv2, Fw2] = compute_RHS(U,V,W,nu,dx,dz,y,ym,yg,Ny,dPdx);
 
U = U0 + dt*(-Fu1+2*Fu2);
V = V0 + dt*(-Fv1+2*Fv2);
W = W0 + dt*(-Fw1+2*Fw2);

% Projection step to correct for incompressibility 
[U, V, W] = compute_projection_step(U,V,W,dx,dz,ym,y,kxx,kzz,Nx,Ny,Nz,DD);

% 3rd RK step
[Fu3, Fv3, Fw3] = compute_RHS(U,V,W,nu,dx,dz,y,ym,yg,Ny,dPdx);
 
U = U0 + dt*(3/4*Fu2+1/4*Fu3);
V = V0 + dt*(3/4*Fv2+1/4*Fv3);
W = W0 + dt*(3/4*Fw2+1/4*Fw3);

% Projection step to correct for incompressibility 
[U, V, W] = compute_projection_step(U,V,W,dx,dz,ym,y,kxx,kzz,Nx,Ny,Nz,DD);

end
