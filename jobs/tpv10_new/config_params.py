#!/usr/bin/env python
import json

#==============================================================================
resample = 1

DH = 100.0/resample
DT = 0.01/2.0/resample
TMAX = 1.0
TMAX += DT

EXPORT_TIME_SKIP = 1 # skip xxx steps to write the snaps

# grid points NX * NY * NZ
NX = 100
NY = 400
NZ = 200

# 0: set in the source code
# 1: input from netcdf files
INPORT_GRID_TYPE = 0
INPORT_STRESS_TYPE = 0

Fault_geometry = 'tpv29_coord_ny906_nz456.nc'
Fault_init_stress = 'Fault_init_stress.nc'

EXPORT_WAVE_SLICE_X = int(NX/2)
EXPORT_WAVE_SLICE_Y = int(NY/2)
EXPORT_WAVE_SLICE_Z = int(NZ-1)

# fault region index (from 1 to N)
#Fault_grid = [51, NY-50, 51, NZ-50] # ny1~ny2 nz1~nz2
Fault_grid = [50, NY-50, 50, NZ] # ny1~ny2 nz1~nz2
src_j = int(NY/2)
src_k = NZ - int(7.5e3/DH)
wid = int(1.5e3/DH) # half width of a square
Asp_grid = [src_j-wid, src_j+wid, src_k-wid, src_k+wid]

smooth_pmax = 0.5
smooth_gauss_width = 0.52

# MPI partition, total MPI process = PX * PY * NZ 
PX = 1; PY = 2; PZ = 1
igpu = 4;
# then each MPI process will be (NX/PX) * (NY/PY) * (NZ/PZ)
# make sure NX, NY, NZ can be divided by PX, PY, PZ, respectively.

# friction law params for slip weakening
mu_s = 0.760  # static friction
mu_d = 0.448 # dynamic friction
Dc = 0.5 # critial distance (m)
C0 = 0.2e6 # Cohesive force (Pa)

#mu_s = 0.12  # static friction
#mu_d = 0.06 # dynamic friction

# friction law params for rate-state weakening
RS_f0 = 0.6
RS_L = 0.02
RS_a0 = 0.01
RS_da0 = 0.01
RS_b0 = 0.01
RS_db0 = 0.01
RS_W = 10e3
RS_w = 3e3
RS_V0 = 1e-6
RS_Vini = 1e-12
Tau_ini = 75e6
dTau_ini = 25e6
Tn_ini = 120e6

# media params
#vp1 = 3000.0
#vs1 = 1500.0
#rho1 = 1200.0
#vp1 = 6000.0
#vs1 = 3464.0
#rho1 = 2670.0
vp1 = 5716.0
vs1 = 3300.0
rho1 = 2700.0

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
PML_N = 12
PML_velocity = 6000
PML_bmax = 2.5
PML_fc = 5.0/3.14;
MPML_pmax = 0.0

# Cerjan
DAMP_N = 50

# out of snaps
tinv_of_seismo = 1
tinv_of_snap = 1
number_of_snap = 2
#id start[3] count[3] strid[3] tinv cmp
#snaps = {
#  'snap_001': {
#    'start': [1, 1, 200],
#    'count': [-1, -1, 1],
#    'stride': [1, 1, 1],
#    'tinv': 1,
#    'cmp': 'V'
#  },
#  'snap_002': {
#    'start': [1, 1, 200],
#    'count': [-1, -1, 1],
#    'stride': [1, 1, 1],
#    'tinv': 1,
#    'cmp': 'V'
#  }
#}
# output directory
OUT = './output'
#OUT = './output_pmax%.2fwidth%.2f' % (smooth_pmax, smooth_gauss_width)
#OUT = './output_pmax%.2fwidth%.2f' % (smooth_pmax, smooth_gauss_width)
#OUT = './output_nonsmooth'
print(OUT)
#==============================================================================

params_dict = {
  'TMAX' : TMAX, 'DT' : DT,
  'DH' : DH,
  'NX' : NX, 'NY' : NY, 'NZ' : NZ,
  'PX' : PX, 'PY' : PY, 'PZ' : PZ,
  'INPORT_GRID_TYPE' : INPORT_GRID_TYPE,
  'INPORT_STRESS_TYPE' : INPORT_STRESS_TYPE,
  'EXPORT_TIME_SKIP' : EXPORT_TIME_SKIP,
  'EXPORT_WAVE_SLICE_X' : EXPORT_WAVE_SLICE_X,
  'EXPORT_WAVE_SLICE_Y' : EXPORT_WAVE_SLICE_Y,
  'EXPORT_WAVE_SLICE_Z' : EXPORT_WAVE_SLICE_Z,
  'igpu' : igpu,
  'Fault_geometry' : Fault_geometry,
  'Fault_init_stress' : Fault_init_stress,
  'Fault_grid' : Fault_grid,
  'Asp_grid' : Asp_grid,
  'smooth_pmax' : smooth_pmax,
  'smooth_gauss_width' : smooth_gauss_width,
  'mu_s' : mu_s,
  'mu_d' : mu_d,
  'Dc' : Dc,
  'C0' : C0,
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
