clc
clear
close all

parfile = 'params.json';
par = get_params(parfile);
NY = par.NY;
NZ = par.NZ;
DT = par.DT;
TMAX = par.TMAX;
TSKP = par.EXPORT_TIME_SKIP;

NT = floor(TMAX/(DT*TSKP));

flag_image = 1

its = 50:50:NT;
its = floor(its);
nt = length(its);

[x,y,z] = gather_coord(parfile);
x = x*1e-3;
y = y*1e-3;
z = z*1e-3;

for i = 1:nt 
it = its(i);
disp(it);
out = par.OUT;
%Vs1 = gather_snap(parfile,out,'Vs1',it);
%Vs2 = gather_snap(parfile,out,'Vs2',it);
Vs1 = gather_snap(parfile,out,'ts1',it);
Vs2 = gather_snap(parfile,out,'ts2',it);
v = sqrt(Vs1.^2+Vs2.^2);

if flag_image
imagesc(y(:,1),z(1,:),v');
else
pcolor(x,y,z,v);
shading interp
end
axis image;axis xy
vm = max(max(abs(v)))/2;
%caxis([0 1]*vm)
colormap( jet )
colorbar
title([num2str(it*DT*TSKP),'s'],'FontSize',12)
set(gca,'FontSize',12)
%axis([-1 1 -1 0]*15)
pause(0.01)
end

%subplot(5,3,15)
%axes('position',[0.2,0.02,.6,.15])
%axis off
%colorbar('south','position',[0.44,0.04,0.15,0.015]);
%caxis([0 5]);
%colorbar('horiz')
%set(gca,'LooseInset',get(gca,'TightInset'))
%set(gca,'looseInset',[0 0 0 0])
%set(gcf, 'PaperPositionMode', 'auto')
%picname = 'tpv102_rough300m'
%picname = 'tpv102_flat'
%if flag_image
%print('-depsc', '-r300', picname)
%else
%print('-dpng', '-r300', picname)
%end
