#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys
import json
from struct import unpack

def usage():
  print(
      '  Usage: ./draw_wave_snap.py -s <slice>\n'+\
      '                             -v <variable>\n'+\
      '                             -t <timestep>\n'+\
      '                             -d <directory>\n'+\
      '     or: ./draw_wave_snap.py --slice <slice>\n'+\
      '                             --var <variable>\n'+\
      '                             --it <timestep>\n'+\
      '                             --dir <directory>\n'+\
      'Default:\n'+\
      '         <slice>     : x (y or z)\n'+\
      '         <variable>  : Vz\n'+\
      '         <timestep>  : 0\n'+\
      '         <directory> : output\n'+\
      'Example:\n'+\
      '         ./draw_wave_snap.py\n'+\
      '         ./draw_wave_snap.py -s x\n'+\
      '         ./draw_wave_snap.py -s x -v Vz\n'+\
      '         ./draw_wave_snap.py -d ouput -v Vx -s x\n'+\
      '         ./draw_wave_snap.py -d ouput -s x -v Vx -t 10\n'+\
      '')

def getOptions(argv):
  import getopt
  slicenm = 'x'
  dirnm = 'output'
  varnm = 'Vz'
  it = 0


  try:
    opts, args = getopt.getopt(argv, "hd:v:t:s:", ["help", "dir=", "var=", "it=", "slice="])
  except getopt.GetoptError:
    print('  Error!')
    usage()
    sys.exit(2)

  for opt, arg in opts:
    if opt in ("-h", "--help"):
      usage()
      sys.exit()
    elif opt in ("-s", "--slice"):
      slicenm = arg
    elif opt in ("-d", "--dir"):
      dirnm = arg
    elif opt in ("-v", "--var"):
      varnm = arg
    elif opt in ("-t", "--it"):
      it = int(arg)

  return dirnm, slicenm, varnm, it

dirnm, slicenm, var, it = getOptions(sys.argv[1:])

print(
    "<<<\n"\
    "   out dir   = %s\n"\
    "   out slice = %s\n"\
    "   var name  = %s\n"\
    "   time step = %d\n"\
    ">>>\n"
    % (dirnm, slicenm, var, it))

par = json.loads(open("params.json").read())
NX = int(par['NX'])
NY = int(par['NY'])
NZ = int(par['NZ'])
PX = int(par['PX'])
PY = int(par['PY'])
PZ = int(par['PZ'])

ni = int(NX / PX)
nj = int(NY / PY)
nk = int(NZ / PZ)

#nx = ni + 6
#ny = nj + 6
#nz = nk + 6

if slicenm == 'x':
  global_i = int(par['EXPORT_WAVE_SLICE_X'])
  i = int(global_i / ni)
  X = np.zeros([NZ, NY])
  Y = np.zeros([NZ, NY])
  Z = np.zeros([NZ, NY])
  V = np.zeros([NZ, NY])
  for j in range(PY):
    for k in range(PZ):

      j1 = j * nj; j2 = j1 + nj
      k1 = k * nk; k2 = k1 + nk

      fnm = '%s/wave_yz_mpi%02d%02d%02d.nc' % (dirnm, i, j, k)
      nc = Dataset(fnm)
      x = nc.variables['x'][:]
      y = nc.variables['y'][:]
      z = nc.variables['z'][:]
      v = nc.variables[var][it,:,:]

      X[k1:k2, j1:j2] = x
      Y[k1:k2, j1:j2] = y
      Z[k1:k2, j1:j2] = z
      V[k1:k2, j1:j2] = v
elif slicenm == 'y':
  global_j = int(par['EXPORT_WAVE_SLICE_Y'])
  j = int(global_j / nj)
  X = np.zeros([NX, NZ])
  Y = np.zeros([NX, NZ])
  Z = np.zeros([NX, NZ])
  V = np.zeros([NX, NZ])
  for i in range(PX):
    for k in range(PZ):

      i1 = i * ni; i2 = i1 + ni
      k1 = k * nk; k2 = k1 + nk

      fnm = '%s/wave_xz_mpi%02d%02d%02d.nc' % (dirnm, i, j, k)
      nc = Dataset(fnm)
      x = nc.variables['x'][:]
      y = nc.variables['y'][:]
      z = nc.variables['z'][:]
      v = nc.variables[var][it,:,:]

      X[i1:i2, k1:k2] = x
      Y[i1:i2, k1:k2] = y
      Z[i1:i2, k1:k2] = z
      V[i1:i2, k1:k2] = v
elif slicenm == 'z':
  global_k = int(par['EXPORT_WAVE_SLICE_Z'])
  k = int(global_k / nk)
  X = np.zeros([NX, NY])
  Y = np.zeros([NX, NY])
  Z = np.zeros([NX, NY])
  V = np.zeros([NX, NY])
  for i in range(PX):
    for j in range(PY):

      i1 = i * ni; i2 = i1 + ni
      j1 = j * nj; j2 = j1 + nj

      fnm = '%s/wave_xy_mpi%02d%02d%02d.nc' % (dirnm, i, j, k)
      nc = Dataset(fnm)
      x = nc.variables['x'][:]
      y = nc.variables['y'][:]
      z = nc.variables['z'][:]
      v = nc.variables[var][it,:,:]

      X[i1:i2, j1:j2] = x
      Y[i1:i2, j1:j2] = y
      Z[i1:i2, j1:j2] = z
      V[i1:i2, j1:j2] = v

vm = np.max(np.abs(V))/2
fig, ax = plt.subplots(1, 1)
#step = 5
#for i in range(0, NX, step):
#  plt.plot(X[i, :], Z[i, :], 'k', linewidth=0.5)
#for k in range(0, NZ, step):
#  plt.plot(X[:, k], Z[:, k], 'k', linewidth=0.5)
#
#i = int(NX/2)
#plt.plot(X[i, :], Z[i, :], 'w', linewidth=1)
cmap='bwr'
cmap='seismic'
#cmap='jet'
if slicenm == 'x':
  plt.pcolormesh(Y, Z, V, vmin=-vm,vmax=vm, cmap=cmap)
  plt.xlabel('y')
  plt.ylabel('z')
elif slicenm == 'y':
  plt.pcolormesh(X, Z, V, vmin=-vm,vmax=vm, cmap=cmap)
  plt.xlabel('x')
  plt.ylabel('z')
  step = 5
  for i in range(0, NX, step):
    plt.plot(X[i,:], Z[i,:], 'k--', linewidth=0.5)
  for k in range(0, NZ, step):
    plt.plot(X[:,k], Z[:,k], 'k--', linewidth=0.5)
elif slicenm == 'z':
  plt.pcolormesh(X, Y, V, vmin=-vm,vmax=vm, cmap=cmap)
  plt.xlabel('x')
  plt.ylabel('y')

plt.title('%s (it=%d)' % (var, it))
plt.colorbar()
plt.axis('image')
plt.show()
