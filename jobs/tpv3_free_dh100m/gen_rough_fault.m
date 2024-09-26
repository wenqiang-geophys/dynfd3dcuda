clc
clear

par = get_params('params.json');

ny = par.NY
nz = par.NZ
dh = par.DH

x = zeros(ny, nz);
y = zeros(ny, nz);
z = zeros(ny, nz);

for j = 1:ny
    for k = 1:nz
        x(j,k) = 0;
        y(j,k) = (j-1-ny/2)*dh;
        z(j,k) = (k-nz)*dh;
    end
end

w = func_selfsimilar2d(ny, 0.1);
w = w(1:ny,1:nz);

x = x + w;
x = x / max(max(abs(x)));


disp('write output...')
fnm_out = ['rough_coord_dh',num2str(dh),'m_2.nc']
ncid = netcdf.create(fnm_out, 'CLOBBER');
dimid(1) = netcdf.defDim(ncid,'ny',ny);
dimid(2) = netcdf.defDim(ncid,'nz',nz);
varid(1) = netcdf.defVar(ncid,'x','NC_FLOAT',dimid);
varid(2) = netcdf.defVar(ncid,'y','NC_FLOAT',dimid);
varid(3) = netcdf.defVar(ncid,'z','NC_FLOAT',dimid);
netcdf.endDef(ncid);
netcdf.putVar(ncid,varid(1),x);
netcdf.putVar(ncid,varid(2),y);
netcdf.putVar(ncid,varid(3),z);
netcdf.close(ncid);

if 1
imagesc(x');axis xy;axis image;
colorbar
colormap('jet')

end

x0=x;
Fs = 1/dh;
[m,n]=size(x0);
T = 1/Fs;
L = m;
t = (0:L-1)*T;
NFFT = 2^nextpow2(L);
Y=fft(x0(:,fix(nz/2)),NFFT)/L;
f=Fs/2*linspace(0,1,NFFT/2+1);
figure;
loglog(f, (abs(Y(1:NFFT/2+1))).^2)
