function w = func_selfsimilar2d(nn,hurst)
if 1
%clear;

lx = nn;       % number of cells lx by lx
lh = lx/2;       % wavenumber for Nyquist frequency

transf = zeros(lx, lx); 

%hurst = 1.0; % Hurst exponent, for self-similar distribution 
exph = hurst + 1.; 

% Generate a self-similar random function, 
ncut = 20; kcut = lh/ncut; % % high-cut filter 
npole = 4; % was 4  
fcsq = kcut*kcut;  % square of filter cutoff wavenumber

% rand('state', 6444);
% rand('state', 644);
if 1
% === Initialize array 'transf' (function in frequency domain) ===
%- coment_by_z for j = 1:lx
%- coment_by_z     for i = 1:lx
%- coment_by_z         % Generate a pair of independent normal random variables
%- coment_by_z         S = 2.;
%- coment_by_z         while S > 1.
%- coment_by_z             vtmp = 2*rand(1,2) - 1; S = vtmp(1)^2 + vtmp(2)^2;
%- coment_by_z         end
%- coment_by_z         transf(i,j) = sqrt(-2*log(S)/S)*sqrt(2.)/2.*complex(vtmp(1),vtmp(2));
%- coment_by_z     end
%- coment_by_z end
A=normrnd(0,1,lx,lx);
B=normrnd(0,1,lx,lx);
transf=sqrt(2.)/2.*complex(A,B);
end
% === Modify array 'transf' in lowest modes, and specify power spectrum ===
for j = 1:lx
    for i = 1:lx
        ik = i-1;
        jk = j-1;
        if (ik > lh); ik = ik-lx; end
        if (jk > lh); jk = jk-lx; end
        ksq = ik^2 + jk^2; 
        fsq = ksq; 
        if ksq == 0
            % Mode (0.0) has zero amplitude
            transf(i,j) = 0.;  
            pspec = 0.;
        elseif ksq == 1
            if jk  == 0
                % Modes (1,0) and (-1,0) have double amplitude
%                transf(i,j) = -2.;
                pspec = 1./fsq^exph;
            else
                % Modes (0,1) and (0,-1) have zero amplitude
%                transf(i,j) = 0.; 
                pspec = 1./fsq^exph;
            end
        elseif ksq == 2
            % Mode (1,1) has zero amplitude
%            transf(i,j) = 0.;
            pspec = 1/fsq^exph;
        else
            % Highers modes have random phase
            pspec = 1/fsq^exph;
        end
        
        filter = 1./(1.+(fsq/fcsq)^npole);
        transf(i,j) = sqrt(pspec*filter)* transf(i,j);
    end
end


% Apply inverse fourier transform
w = real(ifft2(transf)); % 0.25*
w = w/max(abs(w(:)));

if 0
figure; imagesc(w'); xlabel('Lx (strike)'); ylabel('Lz (depth)');
axis image;colormap('jet'); 
end
return
%lambda=80/2;
%[x,y]=meshgrid(0:lx-1,0:lx-1)*lambda;
%
%figure;surf(x,y,w);shading interp;view([0 90]);
end
w=w*1200;

dh=20; %
ds=10; % orginal grid spacing

%y=0:lx-1; yi=0:dh/ds:lx-1;
%z=0:lx-1; zi=0:dh/ds:lx-1;
%x=w(y+1, z+1);
%

y=1:lx; z=1:lx; y = y-0.5; z = z-0.5;
yi=dh/ds*0.5:dh/ds:(lx-dh/ds*0.5);
zi=dh/ds*0.5:dh/ds:(lx-dh/ds*0.5);
x=w(1:lx, 1:lx);

y=y*ds;yi=yi*ds;
z=z*ds;zi=zi*ds;
[Y, Z]=meshgrid(y, z);
[YI, ZI]=meshgrid(yi, zi);
XI=interp2(y,z,x,YI,ZI);

%XI=inpaint_nans(XI);

%figure;imagesc(y, z, x');
figure;imagesc(yi, zi, XI');

if 0
%ny = floor(80e3/dh);
ny = floor(75.6e3/dh);
if ny>length(yi)
ny
  error('ny is larger than lx')
end
nz = floor(22.5e3/dh);
if nz>length(zi)
  error('nz is larger than lx')
end
nx=220;

xgrid=zeros(nz,ny);
ygrid=zeros(nz,ny);
zgrid=zeros(nz,ny);

xgrid=(XI(1:ny, 1:nz))';
for j=1:ny
  ygrid(:,j)=yi(j);
end
for k=1:nz
  zgrid(k,:)=-zi(nz-k+1);
end
% save
end
if 0
fnm_out = 'Geometry_selfsimilar2d.nc';
disp(['To create ', fnm_out])

nc_create_empty(fnm_out);
nc_add_dimension(fnm_out, 'x', nx);
nc_add_dimension(fnm_out, 'y', ny);
nc_add_dimension(fnm_out, 'z', nz);
nc_add_dimension(fnm_out, 'single', 1);

var.Nctype='int';var.Attribute=[];
var.Name='nx';var.Dimension={'single'};nc_addvar(fnm_out,var);
var.Name='ny';var.Dimension={'single'};nc_addvar(fnm_out,var);
var.Name='nz';var.Dimension={'single'};nc_addvar(fnm_out,var);

var.Nctype='float';var.Attribute=[];
var.Name='x';var.Dimension={'z', 'y'};nc_addvar(fnm_out,var);
var.Name='y';var.Dimension={'z', 'y'};nc_addvar(fnm_out,var);
var.Name='z';var.Dimension={'z', 'y'};nc_addvar(fnm_out,var);

nc_varput(fnm_out, 'nx', nx);
nc_varput(fnm_out, 'ny', ny);
nc_varput(fnm_out, 'nz', nz);
nc_varput(fnm_out, 'x', xgrid);
nc_varput(fnm_out, 'y', ygrid);
nc_varput(fnm_out, 'z', zgrid);
disp(['Finished creating "' fnm_out '" file.'])
end

%fnm_nc='Geometry_selfsimilar.nc';
x0=XI;
Fs = 1/dh;
%x0=x;
%Fs = 1/ds;
[m,n]=size(x0);
T = 1/Fs;
L = n;
t = (0:L-1)*T;
NFFT = 2^nextpow2(L);
Y=fft(x0(109,:),NFFT)/L;
f=Fs/2*linspace(0,1,NFFT/2+1);
figure;
loglog(f, (abs(Y(1:NFFT/2+1))).^2)
