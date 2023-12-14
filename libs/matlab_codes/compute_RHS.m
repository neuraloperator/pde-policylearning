function [Fu, Fv, Fw] = compute_RHS(U, V, W, nu, dx, dz, y, ym, yg, Ny, dPdx)
%% Find RHS of the 3D-NS equation
% Fu = -d(uu)/dx -d(uv)/dy -d(uw)/dz + 1/Re*(du/dx + du/dy + du/dz)
% Fv = -d(uv)/dx -d(vv)/dy -d(vw)/dz + 1/Re*(dv/dx + dv/dy + dv/dz)
% Fw = -d(uw)/dx -d(vw)/dy -d(ww)/dz + 1/Re*(dw/dx + dw/dy + dw/dz)

% compute Fu
Fu = zeros(size(U));
% compute -d(uu)/dx
UU = (0.5*(U + U([2:end,1],:,:))).^2;
Fu = Fu - (UU - UU([end,1:end-1],:,:))/dx;
% compute -d(uv)/dy
UV = (0.5*(V + V([end,1:end-1],:,:))).*(0.5*(U(:,1:end-1,:) + U(:,2:end,:)));
for i = 2:Ny
    Fu(:,i,:) = Fu(:,i,:) - (UV(:,i,:) - UV(:,i-1,:))/(y(i)-y(i-1));
end
% compute -d(uw)/dz 
UW = (0.5*(W + W([end,1:end-1],:,:))).*(0.5*(U + U(:,:,[end,1:end-1])));
Fu = Fu - (UW(:,:,[2:end,1]) - UW)/dz;
% compute 1/Re*d^2u/dx^2 
Fu = Fu + nu * ( U([2:end,1],:,:) - 2*U + U([end,1:end-1],:,:) )/dx^2;
% compute 1/Re*d^2u/dz^2
for i = 2:Ny
    Fu(:,i,:) = Fu(:,i,:) + nu * ...
                   (( U(:,i+1,:) - U(:,i  ,:) ) / (yg(i+1)-yg(i  )) - ...
                    ( U(:,i  ,:) - U(:,i-1,:) ) / (yg(i  )-yg(i-1))) / ...
                   (y(i)-y(i-1));
end
% compute 1/Re*d^2u/dz^2
Fu = Fu + nu * ( U(:,:,[2:end,1]) - 2*U + U(:,:,[end,1:end-1]) )/dz^2;

% add pressure gradient
Fu = Fu + dPdx/2;

% compute Fv
Fv = zeros(size(V));
% compute -d(uv)/dx
UV = (0.5*(V + V([end,1:end-1],:,:))).*(0.5*(U(:,1:end-1,:) + U(:,2:end,:)));
Fv = Fv - (UV([2:end,1],:,:) - UV)/dx;
% compute -d(vv)/dy
VV = (0.5*(V(:,1:end-1,:) + V(:,2:end,:))).^2;
for i = 2:Ny-1
    Fv(:,i,:) = Fv(:,i,:) - (VV(:,i,:) - VV(:,i-1,:))/(ym(i)-ym(i-1));
end
% compute -(vw)/dz
VW = (0.5*(V + V(:,:,[end,1:end-1]))).*(0.5*(W(:,1:end-1,:) + W(:,2:end,:)));
Fv = Fv - (VW(:,:,[2:end,1]) - VW)/dz;
% compute nu*d^2v/dx^2
Fv = Fv + nu * ( V([2:end,1],:,:) - 2*V + V([end,1:end-1],:,:) )/dx^2;
for i = 2:Ny-1
    Fv(:,i,:) = Fv(:,i,:) + nu * ...
                   (( V(:,i+1,:) - V(:,i  ,:) ) / (y(i+1)-y(i  )) - ...
                    ( V(:,i  ,:) - V(:,i-1,:) ) / (y(i  )-y(i-1))) / ...
                   (ym(i)-ym(i-1));
end
% compute nu*d^2w/dz^2
Fv = Fv + nu * ( V(:,:,[2:end,1]) - 2*V + V(:,:,[end,1:end-1]) )/dz^2;

% compute Fw
Fw = zeros(size(W));
% compute -d(uw)/dx 
UW = (0.5*(W + W([end,1:end-1],:,:))).*(0.5*(U + U(:,:,[end,1:end-1])));
Fw = Fw - (UW([2:end,1],:,:) - UW)/dx;
% compute -d(vw)/dy
VW = (0.5*(V + V(:,:,[end,1:end-1]))).*(0.5*(W(:,1:end-1,:) + W(:,2:end,:)));
for i = 2:Ny
    Fw(:,i,:) = Fw(:,i,:) - (VW(:,i,:) - VW(:,i-1,:))/(y(i)-y(i-1));
end
% compute -d(ww)/dz
WW = (0.5*(W + W(:,:,[2:end,1]))).^2;
Fw = Fw - (WW - WW(:,:,[end,1:end-1]))/dz;
% compute nu*d^2w/dx^2
Fw = Fw + nu * ( W([2:end,1],:,:) - 2*W + W([end,1:end-1],:,:) )/dx^2;
% compute nu*d^2w/dz^2
for i = 2:Ny
    Fw(:,i,:) = Fw(:,i,:) + nu * ...
                   (( W(:,i+1,:) - W(:,i  ,:) ) / (yg(i+1)-yg(i  )) - ...
                    ( W(:,i  ,:) - W(:,i-1,:) ) / (yg(i  )-yg(i-1))) / ...
                   (y(i)-y(i-1));
end
% compute nu*d^2w/dz^2
Fw = Fw + nu * ( W(:,:,[2:end,1]) - 2*W + W(:,:,[end,1:end-1]) )/dz^2;
end