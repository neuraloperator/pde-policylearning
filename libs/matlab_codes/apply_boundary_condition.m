function [U, V, W] = apply_boundary_condition(U,V,W,Vw1,Vw2);

U(:,  1,:) = -U(:,    2,:);
U(:,end,:) = -U(:,end-1,:);
V(:,  1,:) = Vw1;
V(:,end,:) = -Vw2;  % Zelin add a negative symbol here.
W(:,  1,:) = -W(:,    2,:);
W(:,end,:) = -W(:,end-1,:);

