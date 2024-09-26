clc
clear

parfile = 'params.json';
par = get_params(parfile);
outdir = par.OUT;

y = 0
z = -7.5e3
var = 'Vs1'
[v, t] = get_fault_seismo(parfile,outdir,var,y,z);

figure
plot(t, v)
xlabel('T (sec')
