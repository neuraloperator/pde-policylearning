function [Uout, Vout, Wout] = compute_projection_step(Uin,Vin,Win,dx,dz,ym,y,kxx,kzz,Nx,Ny,Nz,DD)

% global dx dz ym y kxx kzz Nx Ny Nz DD

% compute divergence of velocity field
p = zeros(Nx,Ny-1,Nz);
for i = 1:Ny-1
    p(:,i,:) = (Uin([2:end,1],i+1,:) - Uin(:,i+1,:))/dx + ...
               (Vin(        :,i+1,:) - Vin(:,i  ,:))/(y(i+1)-y(i)) + ...
               (Win(:,i+1,[2:end,1]) - Win(:,i+1,:))/dz;
end

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

% apply fractional step
Uout = Uin; Vout = Vin; Wout = Win;
Uout(:,2:end-1,:) = Uout(:,2:end-1,:) - (p - p([end 1:end-1],:,:))/dx;
for i = 2:Ny-1
    Vout(:,i,:) = Vout(:,i,:) - (p(:,i,:)-p(:,i-1,:))/(ym(i)-ym(i-1));
end
Wout(:,2:end-1,:) = Wout(:,2:end-1,:) - (p - p(:,:,[end 1:end-1]))/dz;

end
