function p = solve_poisson(p, dx, dz, ym, y, kxx, kzz, Nx, Ny, Nz, DD)

% solve poisson equation for p
% Fourier transform
rhs_p_hat = fft(fft(p,[],3),[],1);

% solve Poisson equation in Fourier space
for i = 1:Nx
    for j = 1:Nz
        kk = kxx(i) + kzz(j);       
        D = DD + eye(Ny-1)*kk;
        if (i==1) && (j==1)
            D(1,1) = 1.5*D(1,1);
        end
        rhs_p_hat(i,:,j) = D\conj(squeeze(rhs_p_hat(i,:,j))');
    end
end

% inverse transform
p = real(ifft(ifft(rhs_p_hat,[],1),[],3));