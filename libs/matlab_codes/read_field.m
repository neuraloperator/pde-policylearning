function [x,y,z,xm,ym,zm,U,V,W] = read_field(fin)

fid = fopen(fin,'r','b');

n   = fread(fid,1,'int32');
x   = fread(fid,n,'float64');
    
n   = fread(fid,1,'int32');
y   = fread(fid,n,'float64');
    
n   = fread(fid,1,'int32');
z   = fread(fid,n,'float64');
    
n   = fread(fid,1,'int32');
xm  = fread(fid,n,'float64');
    
n   = fread(fid,1,'int32');
ym  = fread(fid,n,'float64');
    
n   = fread(fid,1,'int32');
zm  = fread(fid,n,'float64');
    
n   = fread(fid,3,'int32');
U   = fread(fid,n(1)*n(2)*n(3),'float64');
U   = reshape(U,n(1),n(2),n(3));
    
n   = fread(fid,3,'int32');
V   = fread(fid,n(1)*n(2)*n(3),'float64');
V   = reshape(V,n(1),n(2),n(3));
    
n   = fread(fid,3,'int32');
W   = fread(fid,n(1)*n(2)*n(3),'float64');
W   = reshape(W,n(1),n(2),n(3));

fclose(fid);

