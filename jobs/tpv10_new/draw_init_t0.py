#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys

#var = 'Init_t0'
#OUT = '../output'
#if (len(sys.argv)>1):
#    OUT = sys.argv[1]
#
#dims = [1,8,4]
#
#NX = 200
#NY = 400
#NZ = 200
#
#nx = NX/dims[0]
#ny = NY/dims[1]
#nz = NZ/dims[2]
#
#X = np.zeros([NZ, NY], dtype='float')
#Y = np.zeros([NZ, NY], dtype='float')
#Z = np.zeros([NZ, NY], dtype='float')
#V = np.zeros([NZ, NY], dtype='float')
#
#n_i = int(dims[0]/2)
#dt = 0.01
#
#for n_j in range(dims[1]):
#  for n_k in range(dims[2]):
#
#    nj1 = int(n_j*ny+0)
#    nj2 = int(n_j*ny+ny)
#    nk1 = int(n_k*nz+0)
#    nk2 = int(n_k*nz+nz)
#
#    fnm = '%s/SnapFault1_1_%02d%02d%02d.nc' % (OUT, n_i, n_j, n_k)
#
#    #print('%s; nj = %4d %4d; nk = %4d %4d' % (fnm, nj1, nj2, nk1, nk2))
#
#    nc = Dataset(fnm)
#    V[nk1:nk2, nj1:nj2] = nc.variables[var][:, :]
#    X[nk1:nk2, nj1:nj2] = nc.variables['x'][:, :]
#    Y[nk1:nk2, nj1:nj2] = nc.variables['y'][:, :]
#    Z[nk1:nk2, nj1:nj2] = nc.variables['z'][:, :]
#
#X = X[50:NZ, 50:NY-50]
#Y = Y[50:NZ, 50:NY-50]
#Z = Z[50:NZ, 50:NY-50]
#V = V[50:NZ, 50:NY-50]
from struct import unpack
import json

par = json.loads(open("./params.json").read())
NX = int(par['NX'])
NY = int(par['NY'])
NZ = int(par['NZ'])
PX = int(par['PX'])
PY = int(par['PY'])
PZ = int(par['PZ'])

DH = float(par['DH'])
DT = float(par['DT'])
TMAX = float(par['TMAX'])

ni = int(NX / PX)
nj = int(NY / PY)
nk = int(NZ / PZ)

nx = ni + 6
ny = nj + 6
nz = nk + 6

OUT = './output'
if (len(sys.argv)>1):
    OUT = sys.argv[1]


X = np.zeros([NZ, NY], dtype='float')
Y = np.zeros([NZ, NY], dtype='float')
Z = np.zeros([NZ, NY], dtype='float')
V = np.zeros([NZ, NY])

i0 = NX/2
i = int(i0 / ni)
for j in range(PY):
  for k in range(PZ):
    j1 = j * nj; j2 = j1 + nj
    k1 = k * nk; k2 = k1 + nk

    fnm = "%s/fault_mpi%02d%02d%02d.nc" % (OUT, i, j, k)
    nc = Dataset(fnm)
    v = nc.variables['init_t0'][:, :]
    #v = np.reshape(data, [nk, nj])
    V[k1:k2, j1:j2] = v
    X[k1:k2, j1:j2] = nc.variables['x'][:, :]
    Y[k1:k2, j1:j2] = nc.variables['y'][:, :]
    Z[k1:k2, j1:j2] = nc.variables['z'][:, :]


X = X[50:NZ, 50:NY-50]
Y = Y[50:NZ, 50:NY-50]
Z = Z[50:NZ, 50:NY-50]
V = V[50:NZ, 50:NY-50]

nc = Dataset('barall/tpv10_init_t0_barall.nc')
Y3 = nc.variables['x'][:]
Z3 = nc.variables['y'][:]
V3 = nc.variables['z'][:, :]
#V3 = V3[::-1, :]

vm = 1
if 1:
    fig, ax = plt.subplots(1, 1)
    vec = np.arange(0.5, 12.0, 0.5)
    vec = np.arange(1.0, 12.0, 1.0)
    plt.contour(Y*1e-3+0.05,-Z*1e-3/np.sin(np.pi/3)-0.05,V, vec, colors='r')
    plt.contour(Y3,Z3,V3, vec, colors='k')
    #ax.invert_yaxis()
    ax.invert_yaxis()
    plt.axis('image')
    plt.xlabel('Along strike distance (km)')
    plt.ylabel('Down dip distance (km)')
    plt.show()

if 1:
    y1 = Y[0, :]
    z1 = Z[:, 0]
    n1, = np.shape(y1)
    n2, = np.shape(z1)
    f = Dataset("tpv10_init_t0_gpu.nc", "w", format='NETCDF4')
    f.createDimension('x', n1)
    f.createDimension('y', n2)
    f.createVariable("x", 'f8', ("x"))
    f.createVariable("y", 'f8', ("y"))
    f.createVariable("z", 'f4', ("y", "x"))
    f.variables["x"][:] = y1*1e-3
    f.variables["y"][:] = -z1*1e-3/np.sqrt(3)*2.
    f.variables["z"][:] = V
    f.close()
