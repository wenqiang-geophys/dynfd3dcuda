#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys
import json

#if(len(sys.argv)<4):
#  print('Usage: %s var index_y index_k' %(sys.argv[0]))
#  exit()

var = sys.argv[1]

#j = int(sys.argv[2]) - 1
#k = int(sys.argv[3]) - 1
ystr = sys.argv[2]
zstr = sys.argv[3]
OUT =  sys.argv[4]

yy = float(ystr)
zz = float(zstr)

DH = 100.0
DT = 0.005 * 1


j = 199 + int(yy*1e3/DH)
k = 199 - int(zz*1e3/DH)

nj = 0
nk = 0

dimx = 0
dimy = 0
dimz = 0
local_j = j - dimy * nj
local_k = k - dimz * nk

print("local( j = %d, k = %d)" % (local_j, local_k))
#fnm = '../output/SnapFault_mpi%02d%02d%02d.nc' % (dimx, dimy, dimz)
fnm = '%s/fault_mpi%02d%02d%02d.nc' % (OUT, dimx, dimy, dimz)
print(fnm)

nc = Dataset(fnm)
v = nc.variables[var][::1,local_k,local_j]

NT = len(v)

t = np.arange(0, NT, 1) * DT

#plt.figure(figsize=(8,4))
if 0:
  fig, ax = plt.subplots(1, 1)
  #fig.set_size_inches(7, 4)
  plt.plot(t, -v, 'k--', linewidth=1.0)
  plt.plot(t0, v0, 'r', linewidth=1.0)
  #plt.title('%s (strike %.1f km; dip %.1f km)' % (var,y/1e3,z/1e3))
  plt.title('%s (j = %d, k = %d)' % (var, j, k))
  #filenm = '%s_strike%s_dip%s.png' % (sys.argv[1], sys.argv[2], sys.argv[3])
  #fig.tight_layout()
  #plt.savefig(filenm)
  plt.show()
outvar = np.zeros([NT, 2])
outvar[:,0] = t
outvar[:,1] = -v
np.savetxt("%s/tpv10_%s_strike%sdip%s_gpu.txt" % (OUT, var, ystr, zstr), outvar)
