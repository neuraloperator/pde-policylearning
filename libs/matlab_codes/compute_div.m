function div = compute_div(U,V,W,dx,y,dz,Nx,Ny,Nz)
div = zeros(Nx,Ny-1,Nz);
for j = 1:Ny-1
    ux = (U([2:end,1],j+1,:) - U(:,j+1,:))/dx;
    uy = (V(:,j+1,:) - V(:,j,:))/(y(j+1)-y(j));
    uz = (W(:,j+1,[2:end,1]) - W(:,j+1,:))/dz;
    div(:,j,:) = ux + uy + uz;
end
end