function snap = gather_snap(parfile, dirnm, var, it)
par = get_params(parfile);

NX = par.NX;
NY = par.NY;
NZ = par.NZ;
DH = par.DH;
PX = par.PX;
PY = par.PY;
PZ = par.PZ;

nj = NY/PY;
nk = NZ/PZ;

snap = zeros(NY,NZ);

%dirnm = par.OUT;

i = 0;
for j = 0:PY-1
for k = 0:PZ-1

fnm = [dirnm, '/fault_mpi',...
num2str(i,'%02d'),...
num2str(j,'%02d'),...
num2str(k,'%02d'),'.nc'];

j1 = j * nj + 1; j2 = j1 + nj-1;
k1 = k * nk + 1; k2 = k1 + nk-1;

if(nargin < 4)
v = ncread(fnm, var, [1 1], [nj nk]);
else
v = ncread(fnm, var, [1 1 it], [nj nk 1]);
v = squeeze(v);
end

snap(j1:j2,k1:k2) = v;

end
end

end
