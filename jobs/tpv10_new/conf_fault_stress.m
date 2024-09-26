clc
clear

par = get_params('params.json');

ny = par.NY
nz = par.NZ
dh = par.DH
OUT = par.OUT

fnm = par.Fault_geometry
x = ncread(fnm, 'x');
y = ncread(fnm, 'y');
z = ncread(fnm, 'z');
vec_n = ncread(fnm, 'vec_n');
vec_m = ncread(fnm, 'vec_m');
vec_l = ncread(fnm, 'vec_l');

%x = imgaussfilt(x, [1 1]*10);
%x = x/max(max(abs(x)));
%x = x * 300;
%if flag_flat
%x(:,:) = 0;
%end

Tx = zeros(ny, nz);
Ty = zeros(ny, nz);
Tz = zeros(ny, nz);

Tn = zeros(ny, nz);
Tm = zeros(ny, nz);
Tl = zeros(ny, nz);

dTx = zeros(ny, nz);
dTy = zeros(ny, nz);
dTz = zeros(ny, nz);

dTn = zeros(ny, nz);
dTm = zeros(ny, nz);
dTl = zeros(ny, nz);

for k = 1:nz
  if(~mod(k,50)) disp([num2str(k), '/', num2str(nz)]); end
  for j = 1:ny

    depth = abs(z(j,k));
    if(depth < 1e-3)
      depth = dh/3;
    end

    %Tn(j,k)=Tx(j,k)*en(1)+Ty(j,k)*en(2)+Tz(j,k)*en(3);
    %Tm(j,k)=Tx(j,k)*em(1)+Ty(j,k)*em(2)+Tz(j,k)*em(3);
    %Tl(j,k)=Tx(j,k)*el(1)+Ty(j,k)*el(2)+Tz(j,k)*el(3);

    %dTn(j,k)=dTx(j,k)*en(1)+dTy(j,k)*en(2)+dTz(j,k)*en(3);
    %dTm(j,k)=dTx(j,k)*em(1)+dTy(j,k)*em(2)+dTz(j,k)*em(3);
    %dTl(j,k)=dTx(j,k)*el(1)+dTy(j,k)*el(2)+dTz(j,k)*el(3);

    Tn(j,k) = -7378.0*(nz-k)*dh;
    if (k == nz) Tn(j,k) = -7378.0*dh/3.0; end
    Tm(j,k) = 0.0;
    Tl(j,k) = 0.55*Tn(j,k);

    y1 = y(j,k);
    z1 = depth/sind(60)-12e3;
    wid = 1.5e3;

    j0 = fix(ny/2);
    k0 = nz-fix(12e3/dh);
    wid = fix(1.5e3/dh);
    if( -wid <= j-j0 && j-j0 < wid && -wid <= k-k0 && k-k0 < wid)
      Tl(j,k) = (0.76+0.0057) * Tn(j,k) - 0.2e6;
    end

    dTn(j,k) = 0;
    dTm(j,k) = 0;
    dTl(j,k) = 0;

    % tranform local n,m,l to global x,y,z axis
    en = squeeze(vec_n(:,j,k));
    em = squeeze(vec_m(:,j,k));
    el = squeeze(vec_l(:,j,k));

    Tx(j,k)=Tn(j,k)*en(1)+Tm(j,k)*em(1)+Tl(j,k)*el(1);
    Ty(j,k)=Tn(j,k)*en(2)+Tm(j,k)*em(2)+Tl(j,k)*el(2);
    Tz(j,k)=Tn(j,k)*en(3)+Tm(j,k)*em(3)+Tl(j,k)*el(3);

    dTx(j,k)=dTn(j,k)*en(1)+dTm(j,k)*em(1)+dTl(j,k)*el(1);
    dTy(j,k)=dTn(j,k)*en(2)+dTm(j,k)*em(2)+dTl(j,k)*el(2);
    dTz(j,k)=dTn(j,k)*en(3)+dTm(j,k)*em(3)+dTl(j,k)*el(3);

  end
end


%% slip weakening friction
mu_s = zeros(ny, nz);
mu_d = zeros(ny, nz);
Dc = zeros(ny, nz);
C0 = zeros(ny, nz);

Fault_grid = [51,350,51,200];

j1 = par.Fault_grid(1)
j2 = par.Fault_grid(2)
k1 = par.Fault_grid(3)
k2 = par.Fault_grid(4)

mu_s(:,:) = 1000;
mu_d(:,:) = 0.448;
Dc(:,:) = 0.5;
C0(:,:) = 1000e6;

mu_s(j1:j2,k1:k2) = 0.76;
C0(j1:j2,k1:k2) = 0.2e6;
%% rate state friction
V0 = 1e-6;
Vini = 1e-16;

W = 15e3; w = 3e3;
By = Bfunc(y, W, w);
Bz = Bfunc(z+7.5e3, W/2, w);
%Bz2 = Bfunc(z+7.5e3, 4.5e3, w);
Bz3 = Bfunc_free(-z, W, w);
%n = fix(w/dh)+1;
%Bz(:,nz-n+1:nz) = Bz2(:,nz-n+1:nz);
B = (1-By.*Bz3);
a = 0.01+0.01*B;
Vw = 0.1+0.9*B;
b = ones(size(a))*0.014;
L = ones(size(a))*0.4;

Tau = sqrt(Tm.^2+Tl.^2);
State = a.*log(2.0*V0/Vini*sinh(Tau./abs(Tn)./a));

B = (1-By.*Bz);
TP_hy = 4e-4 + 1 * B;

fnm_out = par.Fault_init_stress
ncid = netcdf.create(fnm_out, 'CLOBBER');
dimid(1) = netcdf.defDim(ncid,'ny',ny);
dimid(2) = netcdf.defDim(ncid,'nz',nz);
varid(1) = netcdf.defVar(ncid,'x','NC_FLOAT',dimid);
varid(2) = netcdf.defVar(ncid,'y','NC_FLOAT',dimid);
varid(3) = netcdf.defVar(ncid,'z','NC_FLOAT',dimid);
varid(4) = netcdf.defVar(ncid,'Tx','NC_FLOAT',dimid);
varid(5) = netcdf.defVar(ncid,'Ty','NC_FLOAT',dimid);
varid(6) = netcdf.defVar(ncid,'Tz','NC_FLOAT',dimid);
varid(7) = netcdf.defVar(ncid,'dTx','NC_FLOAT',dimid);
varid(8) = netcdf.defVar(ncid,'dTy','NC_FLOAT',dimid);
varid(9) = netcdf.defVar(ncid,'dTz','NC_FLOAT',dimid);
varid(10) = netcdf.defVar(ncid,'mu_s','NC_FLOAT',dimid);
varid(11) = netcdf.defVar(ncid,'mu_d','NC_FLOAT',dimid);
varid(12) = netcdf.defVar(ncid,'Dc','NC_FLOAT',dimid);
varid(13) = netcdf.defVar(ncid,'C0','NC_FLOAT',dimid);
%varid(14) = netcdf.defVar(ncid,'a','NC_FLOAT',dimid);
%varid(15) = netcdf.defVar(ncid,'b','NC_FLOAT',dimid);
%varid(16) = netcdf.defVar(ncid,'L','NC_FLOAT',dimid);
%varid(17) = netcdf.defVar(ncid,'Vw','NC_FLOAT',dimid);
%varid(18) = netcdf.defVar(ncid,'State','NC_FLOAT',dimid);
%varid(19) = netcdf.defVar(ncid,'TP_hy','NC_FLOAT',dimid);
netcdf.endDef(ncid);
netcdf.putVar(ncid,varid(1),x);
netcdf.putVar(ncid,varid(2),y);
netcdf.putVar(ncid,varid(3),z);
netcdf.putVar(ncid,varid(4),Tx);
netcdf.putVar(ncid,varid(5),Ty);
netcdf.putVar(ncid,varid(6),Tz);
netcdf.putVar(ncid,varid(7),dTx);
netcdf.putVar(ncid,varid(8),dTy);
netcdf.putVar(ncid,varid(9),dTz);
netcdf.putVar(ncid,varid(10),mu_s);
netcdf.putVar(ncid,varid(11),mu_d);
netcdf.putVar(ncid,varid(12),Dc);
netcdf.putVar(ncid,varid(13),C0);
%netcdf.putVar(ncid,varid(14),a);
%netcdf.putVar(ncid,varid(15),b);
%netcdf.putVar(ncid,varid(16),L);
%netcdf.putVar(ncid,varid(17),Vw);
%netcdf.putVar(ncid,varid(18),State);
%netcdf.putVar(ncid,varid(19),TP_hy);
netcdf.close(ncid);
