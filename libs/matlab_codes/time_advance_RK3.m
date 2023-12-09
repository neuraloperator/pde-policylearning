function [U, V, W, dPdx] = time_advance_RK3(opV1,opV2, U0,V0,W0,meanU0,nu,dPdx,y,ym,yg,dx,dz,dt,kxx,kzz,Nx,Ny,Nz,DD)
%% Time advance U, V and W using RK3

U  = U0;
V  = V0;
W  = W0;
    
% 1st RK step
[Fu1, Fv1, Fw1] = compute_RHS(U,V,W);

U = U0 + dt*8/15*Fu1;
V = V0 + dt*8/15*Fv1;
W = W0 + dt*8/15*Fw1;

[U, V, W] = apply_boundary_condition(U,V,W);

% Projection step to correct for incompressibility
[U, V, W] = compute_projection_step(U,V,W);

% boundary conditions
[U, V, W] = apply_boundary_condition(U,V,W);

% 2nd RK step
[Fu2, Fv2, Fw2] = compute_RHS(U,V,W);
 
U = U0 + dt*(1/4*Fu1+5/12*Fu2);
V = V0 + dt*(1/4*Fv1+5/12*Fv2);
W = W0 + dt*(1/4*Fw1+5/12*Fw2);

[U, V, W] = apply_boundary_condition(U,V,W);

% Projection step to correct for incompressibility 
[U, V, W] = compute_projection_step(U,V,W);

% boundary conditions
[U, V, W] = apply_boundary_condition(U,V,W);

% 3rd RK step
[Fu3, Fv3, Fw3] = compute_RHS(U,V,W);
 
U = U0 + dt*(1/4*Fu1+3/4*Fu3);
V = V0 + dt*(1/4*Fv1+3/4*Fv3);
W = W0 + dt*(1/4*Fw1+3/4*Fw3);

[U, V, W] = apply_boundary_condition(U,V,W);

% Projection step to correct for incompressibility 
[U, V, W] = compute_projection_step(U,V,W);

% boundary conditions
[U, V, W] = apply_boundary_condition(U,V,W);

dPdx_pre = dPdx;
dPdx = 2*meanU0 - trapz([0; ym; 2],[0,mean(mean(U(:,2:end-1,:),3),1),0]);
U(:,2:end-1,:) = U(:,2:end-1,:) + dPdx/2;
    
dPdx = 0.5*(dPdx_pre+dPdx/dt);

end
