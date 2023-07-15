function dpdx = compute_dpdx_reverse(U, V, W,nu,dx,dz,y,yg,Ny)
%% Find RHS of the 2D-NS equation
% probably should be this:
% Fu = - d(uu)/dx -d(uv)/dy -d(uw)/dz + 1/Re*d^2u/dx^2 + 
%      1/Re*d^2u/dy^2 + 1/Re*d^2u/dz^2
% dpdx = -Fu

% compute Fu
Fu = zeros(size(U));
% compute -d(uu)/dx: (u·∇)u
UU = (0.5*(U + U([2:end,1],:,:))).^2;
Fu = Fu - (UU - UU([end,1:end-1],:,:))/dx;
% compute -d(uv)/dy: 
UV = (0.5*(V + V([end,1:end-1],:,:))).*(0.5*(U(:,1:end-1,:) + U(:,2:end,:)));
for i = 2:Ny
    Fu(:,i,:) = Fu(:,i,:) - (UV(:,i,:) - UV(:,i-1,:))/(y(i)-y(i-1));
end
% compute -d(uw)/dz 
UW = (0.5*(W + W([end,1:end-1],:,:))).*(0.5*(U + U(:,:,[end,1:end-1])));
Fu = Fu - (UW(:,:,[2:end,1]) - UW)/dz;
% compute 1/Re*d^2u/dx^2 
Fu = Fu + nu * (U([2:end,1],:,:) - 2*U + U([end,1:end-1],:,:))/dx^2;
% compute 1/Re*d^2u/dy^2
for i = 2:Ny
    Fu(:,i,:) = Fu(:,i,:) + nu * ...
                   (( U(:,i+1,:) - U(:,i  ,:) ) / (yg(i+1)-yg(i  )) - ...
                    ( U(:,i  ,:) - U(:,i-1,:) ) / (yg(i  )-yg(i-1))) / ...
                   (y(i)-y(i-1));
end
% compute 1/Re*d^2u/dz^2
Fu = Fu + nu * ( U(:,:,[2:end,1]) - 2*U + U(:,:,[end,1:end-1]) )/dz^2;
dpdx = -Fu;
end

