#!/usr/bin/env python
import json
import os

#==============================================================================
resample = 1

DH = 100.0/resample
DT = 0.016/2.0/resample
TMAX = 12.0

TMAX += DT

EXPORT_TIME_SKIP = resample # skip xxx steps to write the snaps

# grid points NX * NY * NZ
NX = 120
NN = int(0e3/DH) + int(50*resample)
NY = int(30e3/DH) + NN * 2
NZ = int(15e3/DH) + NN

# Input grid and init stress
# 0: set in the source code
# 1: input from netcdf files
INPORT_GRID_TYPE = 1
INPORT_STRESS_TYPE = 1
Fault_init_stress = 'init_stress.nc'
Fault_geometry = 'fault_coord.nc'

# Friction type
# 0 slip-weakening
# 1 rate-state with ageing law
# 2 rate-state with slip law
Friction_type = 0 

EXPORT_WAVE_SLICE_X = int(NX/2)
EXPORT_WAVE_SLICE_Y = int(NY/2)
EXPORT_WAVE_SLICE_Z = int(NZ-1)

# fault region index (from 1 to N)
#Fault_grid = [51, NY-50, 51, NZ-50] # ny1~ny2 nz1~nz2
Fault_grid = [NN, NY-NN, NN, NZ] # ny1~ny2 nz1~nz2
src_j = int(NY/2)
src_k = NZ - int(7.5e3/DH)
wid = int(1.5e3/DH) # half width of a square
Asp_grid = [src_j-wid, src_j+wid, src_k-wid, src_k+wid]

# MPI partition, total MPI process = PX * PY * NZ 
PX = 1; PY = 4; PZ = 2
if resample == 1:
  PX = 1; PY = 1; PZ = 1
igpu = 5;
# then each MPI process will be (NX/PX) * (NY/PY) * (NZ/PZ)
# make sure NX, NY, NZ can be divided by PX, PY, PZ, respectively.

# friction law params for slip weakening
mu_s = 0.677  # static friction
mu_d = 0.525 # dynamic friction
Dc = 0.4 # critial distance (m)
C0 = 0.0e6 # Cohesive force (Pa)

#mu_s = 0.12  # static friction
#mu_d = 0.06 # dynamic friction

# friction law params for rate-state weakening
RS_f0 = 0.6
RS_fw = 0.2 # only used for RS(slip law)
RS_V0 = 1e-6
RS_Vini = 1e-12
RS_L = 0.02
#RS_a0 = 0.01
#RS_da0 = 0.01
#RS_b0 = 0.01
#RS_db0 = 0.01
#RS_W = 10e3
#RS_w = 3e3
#Tau_ini = 75e6
#dTau_ini = 25e6
#Tn_ini = 120e6

smooth_load_T = 1.0

# media params
#vp1 = 3000.0
#vs1 = 1500.0
#rho1 = 1200.0
vp1 = 6000.0
vs1 = 3464.0
rho1 = 2670.0

bi_vp1 = vp1/1.0
bi_vs1 = vs1/1.0
bi_rho1 = rho1/1.0

bi_vp2 = vp1/1.0
bi_vs2 = vs1/1.0
bi_rho2 = rho1/1.0

bi_vp1 = vp1
bi_vs1 = vs1
bi_rho1 = rho1

# PML
PML_N = 20
PML_velocity = vp1
PML_bmax = 3.0
PML_fc = 1.0;
MPML_pmax = 0.0

# Cerjan
DAMP_N = 50

# out of snaps
tinv_of_seismo = 1
tinv_of_snap = 1
number_of_snap = 2

# input and output directory
OUT = '/shdisk/lab7/wqzhang/tpv3_free/output_dh%d_tmp' % DH
cmd = 'mkdir -p %s' % (OUT)
os.system(cmd)

Fault_geometry = OUT + '/' + Fault_geometry
Fault_init_stress = OUT + '/' + Fault_init_stress
#==============================================================================

params_dict = {
  'TMAX' : TMAX, 'DT' : DT,
  'DH' : DH,
  'NX' : NX, 'NY' : NY, 'NZ' : NZ,
  'PX' : PX, 'PY' : PY, 'PZ' : PZ,
  'INPORT_GRID_TYPE' : INPORT_GRID_TYPE,
  'INPORT_STRESS_TYPE' : INPORT_STRESS_TYPE,
  'Friction_type' : Friction_type,
  'EXPORT_TIME_SKIP' : EXPORT_TIME_SKIP,
  'EXPORT_WAVE_SLICE_X' : EXPORT_WAVE_SLICE_X,
  'EXPORT_WAVE_SLICE_Y' : EXPORT_WAVE_SLICE_Y,
  'EXPORT_WAVE_SLICE_Z' : EXPORT_WAVE_SLICE_Z,
  'igpu' : igpu,
  'Fault_geometry' : Fault_geometry,
  'Fault_init_stress' : Fault_init_stress,
  'Fault_grid' : Fault_grid,
  'Asp_grid' : Asp_grid,
  'mu_s' : mu_s,
  'mu_d' : mu_d,
  'Dc' : Dc,
  'C0' : C0,
  'RS_V0' : RS_V0,
  'RS_Vini' : RS_Vini,
  'RS_f0' : RS_f0,
  'RS_fw' : RS_fw,
  'RS_L' : RS_L,
  'smooth_load_T' : smooth_load_T,
  'PML_N' : PML_N,
  'PML_velocity' : PML_velocity,
  'PML_bmax' : PML_bmax,
  'PML_fc' : PML_fc,
  'MPML_pmax' : MPML_pmax,
  'DAMP_N' : DAMP_N,
  'vp1': vp1, 'vs1': vs1, 'rho1': rho1,
  'bi_vp1' : bi_vp1, 'bi_vs1' : bi_vs1, 'bi_rho1' : bi_rho1,
  'bi_vp2' : bi_vp2, 'bi_vs2' : bi_vs2, 'bi_rho2' : bi_rho2,
  'OUT' : OUT
}
#json.dumps(test_dict)

json_str = json.dumps(params_dict, indent=4)
#print(json_str)
with open("./params.json", "w") as f:
  #json.dump(test_dict, f)
  f.write(json_str)
  print("configure json finished!")
