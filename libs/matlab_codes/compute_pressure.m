function P = compute_pressure(U,V,W,nu,dPdx,y,ym,yg,dx,dz,kxx,kzz,Nx,Ny,Nz,DD)

% compute pressure term by solving Poisson equation
% compute the RHS of the NS eq
[RHS_u, RHS_v, RHS_w] = compute_RHS(U,V,W,nu,dx,dz,y,ym,yg,Ny,dPdx);

% RHS_p is computed at cell center
% compute the RHS of the pressure Poisson equation by taking the divergence
RHS_p  = zeros([Nx,Ny-1,Nz]);
for i = 1:Ny-1
    RHS_p(:,i,:) = (RHS_u([2:end,1],i+1,:) - RHS_u(:,i+1,:))/dx + ...
                   (RHS_v(        :,i+1,:) - RHS_v(:,i  ,:))/(y(i+1)-y(i)) + ...
                   (RHS_w(:,i+1,[2:end,1]) - RHS_w(:,i+1,:))/dz;
end
                           
% Fourier transform in x and z
RHS_p_hat  = fft2( permute(RHS_p,[1,3,2]) );

for i = 1:Nx
    for j = 1:Nz
        kk = kxx(i) + kzz(j);       
        D = DD + eye(Ny-1)*kk;
        if (i==1) && (j==1)
            D(1,1) = 1.5*D(1,1);
        end
        RHS_p_hat(i,j,:) = D\squeeze(RHS_p_hat(i,j,:));
    end
end

% Transform back to physical space to get pressure
P = real(permute(ifft2(RHS_p_hat),[1,3,2]));
