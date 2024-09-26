clc
clear

prefix = 'rough'

par = get_params('params.json');
dh = par.DH

fnm = par.Fault_geometry

x0 = ncread(fnm, 'x');
y0 = ncread(fnm, 'y');
z0 = ncread(fnm, 'z');
[ny0, nz0] = size(x0);
ny = 2*ny0;
nz = 2*nz0;

i1 = linspace(0,1,ny0);
j1 = linspace(0,1,nz0);

i2 = linspace(0,1,ny);
j2 = linspace(0,1,nz);

[i1,j1]=meshgrid(i1,j1);
[i2,j2]=meshgrid(i2,j2);

x = interp2(i1,j1,x0',i2,j2)';
y = interp2(i1,j1,y0',i2,j2)';
z = interp2(i1,j1,z0',i2,j2)';


disp('calculating metric and base vectors...')
metric = cal_metric(x,y,z, dh);
[vec_n, vec_m, vec_l] = cal_basevectors(metric);


if 0
u = squeeze(vec_n(1,:,:))*1e3;
v = squeeze(vec_n(2,:,:))*1e3;
w = squeeze(vec_n(3,:,:))*1e3;
s=10;
j=1:s:ny;
k=1:s:nz;
figure
quiver3(x(j,k),y(j,k),z(j,k),u(j,k),v(j,k),w(j,k));
end

dh = dh/2;
disp('write output...')
fnm_out = 'fault_coord_refine.nc'
ncid = netcdf.create(fnm_out, 'CLOBBER');
dimid(1) = netcdf.defDim(ncid,'ny',ny);
dimid(2) = netcdf.defDim(ncid,'nz',nz);
dimid3(1) = netcdf.defDim(ncid, 'dim', 3);
dimid3(2) = dimid(1);
dimid3(3) = dimid(2);
varid(1) = netcdf.defVar(ncid,'x','NC_FLOAT',dimid);
varid(2) = netcdf.defVar(ncid,'y','NC_FLOAT',dimid);
varid(3) = netcdf.defVar(ncid,'z','NC_FLOAT',dimid);
varid(4) = netcdf.defVar(ncid,'vec_n','NC_FLOAT',dimid3);
varid(5) = netcdf.defVar(ncid,'vec_m','NC_FLOAT',dimid3);
varid(6) = netcdf.defVar(ncid,'vec_l','NC_FLOAT',dimid3);
netcdf.endDef(ncid);
netcdf.putVar(ncid,varid(1),x);
netcdf.putVar(ncid,varid(2),y);
netcdf.putVar(ncid,varid(3),z);
netcdf.putVar(ncid,varid(4),vec_n);
netcdf.putVar(ncid,varid(5),vec_m);
netcdf.putVar(ncid,varid(6),vec_l);
netcdf.close(ncid);
