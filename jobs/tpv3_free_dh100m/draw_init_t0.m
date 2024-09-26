clc
clear


parfile = 'params.json';
par = get_params(parfile);
NY = par.NY;
NZ = par.NZ;
DT = par.DT;
DH = par.DH;
TMAX = par.TMAX;
TSKP = par.EXPORT_TIME_SKIP;

NT = floor(TMAX/(TSKP*DT));

[x,y,z] = gather_coord(parfile);
x = x * 1e-3;
y = y * 1e-3;
z = z * 1e-3;
y1 = y(:,1);
z1 = z(1,:)';

t = gather_snap(parfile,par.OUT,'init_t0');
v = cal_rup_v(t, DH);
v = v / 3464;
v(v<0) = nan;
v(v>2) = nan;

vec = 0.5:0.5:10;

figure
%imagesc(y1,z1,v');
pcolor(y,z,v);shading interp
caxis([0.5 1.5])
hold on
contour(y, z, t, vec, 'color', 'k', 'linewidth', 1.5)
hold off
title('rupture velocity / Vs')
colorbar

axis image;axis xy
%colormap( coolwarm )
colormap( jet )
axis([-18 18 -18 0])

ylabel('Down-dip (km)')
xlabel('Along-strike (km)')

set(gcf,'PaperPositionMode', 'auto')
%print('-depsc', '-painters', 'tpv102_rupture_time_rough')
