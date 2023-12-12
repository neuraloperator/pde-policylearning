function STA = main_run(x,y,z,xm,ym,zm,UU,VV,WW)
% global nu dPdx y ym yg dx dz dt kxx kzz Nx Ny Nz DD s tempa
% Load data from channel180_minchan.init
% [x,y,z,xm,ym,zm,UU,VV,WW] = read_field('data/channel180_minchan.init');
format long;
% Save data to a .mat file

% Parameters
%----------------------------------------------%
STA = 0;  % status
nu   = 1/0.32500000E+04; % kinematic viscosity 
dPdx = 0.57231059E-01^2; % pressure gradient (utau^2)
%Nx   = 32;  % number of points in the x direction
%Ny   = 130; % number of points in the y direction
%Nz   = 32;  % number of poitns in the z direction
Nt   = 10; % number of time steps
dt   = 0.001; % time step
%----------------------------------------------%
% define grid spacing
%dx = 2*pi/Nx;
%dz = 2*pi/Nz;
%y  = 1 + tanh(2.6*linspace(-1,1,Ny))/tanh(2.6);
%ym = 0.5*(y(1:end-1) + y(2:end));
%yg = [-ym(1), ym, 2+ym(1)];

% initialize with saved field
% [x,y,z,xm,ym,zm,UU,VV,WW] = read_field('channel180_minchan.init');
dx = x(2) - x(1);
dz = z(2) - z(1);
yg = [-ym(1); ym(:); 2+ym(1)];
Nx = length(x)-2;
Nz = length(z)-2;
Ny = length(y);

U = UU(1:Nx  ,:,2:Nz+1);
V = VV(2:Nx+1,:,2:Nz+1);
W = WW(2:Nx+1,:,1:Nz  );

rng('default')
s = rng;
%----------------------------------------------%
% define modified wavenumbers
kxx = zeros(Nx,1); % -kx^2
kzz = zeros(Nz,1); % -ky^2
for k=0:Nx/2
    kxx(k+1) = 2*( cos(2*pi*k/Nx) - 1 )/dx^2;
end
for k=(Nx/2+1):(Nx-1)
    kxx(k+1) = 2*( cos(2*pi*(-Nx+k)/Nx) - 1 )/dx^2;
end

for k=0:Nz/2
    kzz(k+1) = 2*( cos(2*pi*k/Nz) - 1 )/dz^2;
end
for k=(Nz/2+1):(Nz-1)
    kzz(k+1) = 2*( cos(2*pi*(-Nz+k)/Nz) - 1 )/dz^2;
end
%----------------------------------------------%
% define matrix for y direction Poisson solver
DD = zeros(Ny-1,Ny-1);
for j = 1:Ny-1
    DD(j,j) = -1/(y(j+1)-y(j))*(1/(yg(j+2) - yg(j+1)) + 1/(yg(j+1) - yg(j)));
end

for j = 1:Ny-2
    DD(j+1,j) = 1/( y(j+2)-y(j+1) )/( yg(j+2) - yg(j+1) );
    DD(j,j+1) = 1/( y(j+1)-y(j) )/( yg(j+2) - yg(j+1) );
end

DD(1,1)     = DD(1,1) + 1/( y(2)-y(1) )/( yg(2) - yg(1) );
DD(end,end) = DD(end,end) + 1/( y(Ny)-y(Ny-1) )/( yg(Ny+1) - yg(Ny) );
%----------------------------------------------%

% main time loop
for i = 1:Nt
       
    
    % display divergence
    div = zeros(Nx,Ny-1,Nz);
    uxsum = 0;
    uysum = 0;
    uzsum = 0;
    for j = 1:Ny-1
        ux = (U([2:end,1],j+1,:) - U(:,j+1,:))/dx;
        uy = (V(:,j+1,:) - V(:,j,:))/(y(j+1)-y(j));
        uz = (W(:,j+1,[2:end,1]) - W(:,j+1,:))/dz;
        div(:,j,:) = ux + uy + uz;
%         disp(sum(abs(ux(:))));
%         disp(sum(uy(:)));
%         disp(sum(uz(:)));
        uxsum = uxsum + abs(ux);
        uysum = uysum + abs(uy);
        uzsum = uzsum + abs(uz);
    end
    disp(sum(div(:)));
%     disp(mean(abs(U(:))));
%     disp(mean(abs(V(:))));
%     disp(mean(abs(W(:))));
    % advance in time (RK3)
    P  = compute_pressure(U,V,W,nu,dPdx,y,ym,yg,dx,dz,kxx,kzz,Nx,Ny,Nz,DD);
    opV2 = compute_opposition(squeeze(-0.5*(P(:,end,:)+P(:,end-1,:))));
    [U, V, W] = time_advance_RK3(opV2, U, V, W, nu,dPdx,y,ym,yg,dx,dz,dt,kxx,kzz,Nx,Ny,Nz,DD);
%     disp(mean(abs(U(:))));
%     disp(mean(abs(V(:))));
%     disp(mean(abs(W(:))));
%     str = sprintf(['%.', num2str(precision), 'f'], sum(V(:)));
%     disp(str);
STA=1;
end
