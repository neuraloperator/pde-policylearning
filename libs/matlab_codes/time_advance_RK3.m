function [U, V, W, dPdx] = time_advance_RK3(opV1,opV2, U0,V0,W0,meanU0,nu,dPdx,y,ym,yg,dx,dz,dt,kxx,kzz,Nx,Ny,Nz,DD)
%% Time advance U, V and W using RK3

U  = U0;
V  = V0;
W  = W0;

opV1 = opV1 - mean(mean(opV1));
opV2 = opV2 - mean(mean(opV2));

% boundary conditions
[U0, V0, W0] = apply_boundary_condition(U0,V0,W0,-opV1,-opV2);

% 1st RK step
[Fu1, Fv1, Fw1] = compute_RHS(U,V,W,nu,dx,dz,y,ym,yg,Ny,dPdx);

U = U0 + dt*1/3*Fu1;
V = V0 + dt*1/3*Fv1;
W = W0 + dt*1/3*Fw1;

% Projection step to correct for incompressibility
[U, V, W] = compute_projection_step(U,V,W,dx,dz,ym,y,kxx,kzz,Nx,Ny,Nz,DD);

% boundary conditions
[U, V, W] = apply_boundary_condition(U,V,W,-opV1,-opV2);

% 2nd RK step
[Fu2, Fv2, Fw2] = compute_RHS(U,V,W,nu,dx,dz,y,ym,yg,Ny,dPdx);
 
U = U0 + dt*(-Fu1+2*Fu2);
V = V0 + dt*(-Fv1+2*Fv2);
W = W0 + dt*(-Fw1+2*Fw2);

% Projection step to correct for incompressibility 
[U, V, W] = compute_projection_step(U,V,W,dx,dz,ym,y,kxx,kzz,Nx,Ny,Nz,DD);

% boundary conditions
[U, V, W] = apply_boundary_condition(U,V,W,-opV1,-opV2);

% 3rd RK step
[Fu3, Fv3, Fw3] = compute_RHS(U,V,W,nu,dx,dz,y,ym,yg,Ny,dPdx);
 
U = U0 + dt*(3/4*Fu2+1/4*Fu3);
V = V0 + dt*(3/4*Fv2+1/4*Fv3);
W = W0 + dt*(3/4*Fw2+1/4*Fw3);

% Projection step to correct for incompressibility 
[U, V, W] = compute_projection_step(U,V,W,dx,dz,ym,y,kxx,kzz,Nx,Ny,Nz,DD);

% boundary conditions
[U, V, W] = apply_boundary_condition(U,V,W,-opV1,-opV2);

dPdx = meanU0 - trapz([0; ym; 2],[0,mean(mean(U(:,2:end-1,:),3),1),0])/2;
U(:,2:end-1,:) = U(:,2:end-1,:) + dPdx;
[U, V, W] = apply_boundary_condition(U,V,W,-opV1,-opV2);
dPdx = dPdx/dt;
end
