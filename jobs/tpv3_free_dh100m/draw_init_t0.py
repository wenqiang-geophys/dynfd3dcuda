#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from netCDF4 import Dataset
import sys
import json
#from struct import unpack

def usage():
  print(
      '  Usage: ./draw_fault_snap.py -d <directory>\n'+\
      '                              -v <variable>\n'+\
      '                              -t <timestep>\n'+\
      '                              -c\n'+\
      '     or: ./draw_fault_snap.py --dir <directory>\n'+\
      '                              --var <variable>\n'+\
      '                              --it <timestep>\n'+\
      '                              --cut\n'+\
      'Default:\n'+\
      '         <directory> : output\n'+\
      '         <variable>  : ts1\n'+\
      '         <timestep>  : 0\n'+\
      '         -c          : do not cut fault region\n'+\
      'Example:\n'+\
      '         ./draw_fault_snap.py\n'+\
      '         ./draw_fault_snap.py -v Vs1\n'+\
      '         ./draw_fault_snap.py -d ouput -v Vs1\n'+\
      '         ./draw_fault_snap.py -d ouput -v Vs1 -t 10\n'+\
      '         ./draw_fault_snap.py -d ouput -v Vs1 -t 10 -c\n'+\
      '')


def getOptions(argv):
  import getopt
  dirnm = 'output'
  varnm = 'ts1'
  it = 0
  flag_cut = 1

  try:
    opts, args = getopt.getopt(argv, "hd:v:t:c", ["help", "dir=", "var=", "it="])
  except getopt.GetoptError:
    print('  Error!')
    usage()
    sys.exit(2)

  for opt, arg in opts:
    if opt in ("-h", "--help"):
      usage()
      sys.exit()
    elif opt in ("-d", "--dir"):
      dirnm = arg
    elif opt in ("-v", "--var"):
      varnm = arg
    elif opt in ("-t", "--it"):
      it = int(arg)
    elif opt in ("-c", "--cut"):
      flag_cut = 0

  return dirnm, varnm, it, flag_cut

dirnm, var, it, flag_cut = getOptions(sys.argv[1:])

print(
    "<<<\n"\
    "   out dir   = %s\n"\
    "   var name  = %s\n"\
    "   time step = %d\n"\
    "   cut fault = %d\n"\
    ">>>\n"
    % (dirnm, var, it, flag_cut))

par = json.loads(open("params.json").read())
NX = int(par['NX'])
NY = int(par['NY'])
NZ = int(par['NZ'])
PX = int(par['PX'])
PY = int(par['PY'])
PZ = int(par['PZ'])
TIME_SKIP = int(par['EXPORT_TIME_SKIP'])
DT = float(par['DT'])
DH = float(par['DH'])

ni = int(NX / PX)
nj = int(NY / PY)
nk = int(NZ / PZ)

nx = ni + 6
ny = nj + 6
nz = nk + 6

V = np.zeros([NZ, NY])
X = np.zeros([NZ, NY])
Y = np.zeros([NZ, NY])
Z = np.zeros([NZ, NY])
i0 = NX/2
i = int(i0 / ni)

def gather_coord():
  X = np.zeros([NZ, NY])
  Y = np.zeros([NZ, NY])
  Z = np.zeros([NZ, NY])
  for j in range(PY):
    for k in range(PZ):
      j1 = j * nj; j2 = j1 + nj
      k1 = k * nk; k2 = k1 + nk
      fnm = '%s/fault_mpi%02d%02d%02d.nc' % (dirnm, i, j, k)
      nc = Dataset(fnm)
      x = nc.variables['x'][:,:]
      y = nc.variables['y'][:,:]
      z = nc.variables['z'][:,:]
      X[k1:k2, j1:j2] = x
      Y[k1:k2, j1:j2] = y
      Z[k1:k2, j1:j2] = z
  return X, Y, Z

X, Y, Z = gather_coord()

for j in range(PY):
  for k in range(PZ):
    j1 = j * nj; j2 = j1 + nj
    k1 = k * nk; k2 = k1 + nk

    fnm = '%s/fault_mpi%02d%02d%02d.nc' % (dirnm, i, j, k)
    nc = Dataset(fnm)
    #x = nc.variables['x'][:]
    y = nc.variables['y'][:]
    z = nc.variables['z'][:]

    if var in ("Vs1", "Vs2", "ts1", "ts2", "tn", "Us0", "State", "rup_index_y", "rup_index_z"):
      v = nc.variables[var][it,:,:]
    elif(var == 'rup_v'):
        v = nc.variables['init_t0'][:,:]
    else:
        v = nc.variables[var][:,:]

    V[k1:k2, j1:j2] = v
    Y[k1:k2, j1:j2] = y
    Z[k1:k2, j1:j2] = z

Fault_grid = par['Fault_grid']
j1 = int(Fault_grid[0]-1)
j2 = int(Fault_grid[1]-1)
k1 = int(Fault_grid[2]-1)
k2 = int(Fault_grid[3]-1)

if flag_cut:
  V = V[k1:k2, j1:j2]
#print(np.shape(V))

def cal_rup_v(t, steph):
  m, n = np.shape(t)
  v  = np.zeros([m, n], dtype=float)
  sx = np.zeros([m, n], dtype=float)
  sy = np.zeros([m, n], dtype=float)
  d = 4
  #sx(i,j)=(t(i+d,j+d)-t(i-d,j+d)+t(i+d,j-d)-t(i-d,j-d))/4/steph/d;
  #sy(i,j)=(t(i+d,j+d)-t(i+d,j-d)+t(i-d,j+d)-t(i-d,j-d))/4/steph/d;
  #i = np.arange(d,m-d)
  #j = np.arange(d,n-d)
  for i in range(d,m-d):
    for j in range(d,n-d):
      sx[i,j]=(t[i+d,j]-t[i-d,j])/(2*d)
      sy[i,j]=(t[i,j+d]-t[i,j-d])/(2*d)
  for i in range(0,d):
    for j in range(d,n-d):
      sx[i,j]=(t[i+d,j]-t[i,j])/(1*d)
      sy[i,j]=(t[i,j+d]-t[i,j-d])/(2*d)
  for i in range(m-d,m):
    for j in range(d,n-d):
      sx[i,j]=(t[i,j]-t[i-d,j])/(1*d)
      sy[i,j]=(t[i,j+d]-t[i,j-d])/(2*d)
  for i in range(d,m-d):
    for j in range(0,d):
      sx[i,j]=(t[i+d,j]-t[i-d,j])/(2*d)
      sy[i,j]=(t[i,j+d]-t[i,j])/(1*d)
  for i in range(d,m-d):
    for j in range(n-d,n):
      sx[i,j]=(t[i+d,j]-t[i-d,j])/(2*d)
      sy[i,j]=(t[i,j]-t[i,j-d])/(1*d)
  s = np.sqrt(sx**2+sy**2)
  v = steph/(s+1e-6)
  return v
#vm = 1
#V = np.abs(V)
#V = -V
#v = v.transpose()
#v = np.log10(np.abs(v))

if 1:
  #if var == 'ts1':
  #  #V = V - (-40e6)
  #  V = V[21:-21,21:-21]
  fig, ax = plt.subplots(1, 1)
  if var == 'init_t0':
    V[np.abs(V)>999] = -9999.9
    vm = np.max(V)
    print('vm = ', vm)
    vec = np.arange(0.5, vm, 0.5)
    #plt.contour(V, vec, colors='k')
    #plt.contour(V, vec, cmap='viridis')
    plt.contour(V, vec, colors='r')
  elif var == 'rup_v':
    rupv = cal_rup_v(V, DH)
    Vs = 3464.0
    rupv = rupv/Vs
    rupv[rupv>2] = 0
    vm = np.max(rupv)
    #plt.imshow(rupv, origin='lower', vmin=0.5, vmax=1.5, cmap='coolwarm')
    plt.imshow(Y, Z, rupv, origin='lower', vmin=0.5, vmax=1.5, cmap='coolwarm')
    plt.colorbar(shrink=1)
    vm = np.max(V)
    vec = np.arange(0.5, vm, 0.5)
    #plt.contour(Y, Z, V, vec, colors='k',linewidth=0.5)
  else:
    #V = np.log10(np.abs(V))
    plt.imshow(V, origin='lower', cmap='jet')
    plt.colorbar(shrink=0.53)
    #plt.pcolormesh(Y, Z, V, cmap='jet') 
    # jet RdBu bwr
      #vmin=0,vmax=2.0,
      #norm=colors.LogNorm(vmin=1e-1*V.max(),vmax=V.max()),
      #vmin=-vm/2.0,vmax=vm/2.0,
  #ax.invert_yaxis()
  plt.xlabel('y')
  plt.ylabel('z')
  plt.title('%s @ t(%d) = %g sec' % (var, it, it*DT*TIME_SKIP))
  plt.axis('image')
  plt.show()

if 0:
  y1 = y[0, :]
  z1 = z[:, 0]
  n1, = np.shape(y1)
  n2, = np.shape(z1)
  f = Dataset("snap_%s_it%d.nc" % (var, it), "w", format='NETCDF4')
  f.createDimension('x', n1)
  f.createDimension('y', n2)
  f.createVariable("x", 'f8', ("x"))
  f.createVariable("y", 'f8', ("y"))
  f.createVariable("z", 'f4', ("y", "x"))
  f.variables["x"][:] = y1*1e-3
  f.variables["y"][:] = -z1*1e-3
  f.variables["z"][:] = V
  f.close()
