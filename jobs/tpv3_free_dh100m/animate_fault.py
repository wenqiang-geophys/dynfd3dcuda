#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys
import json
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

FONT_SIZE = 12

plt.rc('font', size = FONT_SIZE)
plt.rc('axes', titlesize = FONT_SIZE + 2)
plt.rc('axes', labelsize = FONT_SIZE + 1)
plt.rc('xtick', labelsize = FONT_SIZE)
plt.rc('ytick', labelsize = FONT_SIZE)
plt.rc('legend', fontsize = FONT_SIZE)
plt.rc('figure', titlesize = FONT_SIZE + 4)

def usage():
  print(
      '  Usage: ./animate_fault.py -v <variable>\n'+\
      '                            -t <tskip>\n'+\
      '                            -d <directory>\n'+\
      '     or: ./animate_fault.py --var   <variable>\n'+\
      '                            --tskip <tskip>\n'+\
      '                            --dir   <directory>\n'+\
      'Default:\n'+\
      '         <variable>  : Vz\n'+\
      '         <tskip>     : 1 (2 or 10 ... to draw faster)\n'+\
      '         <directory> : output\n'+\
      'Example:\n'+\
      '         ./animate_fault.py\n'+\
      '         ./animate_fault.py -v Vs1\n'+\
      '         ./animate_fault.py -d ouput -v Vs1 -t 10\n'+\
      '')

def getOptions(argv):
  import getopt
  dirnm = 'output'
  varnm = 'Vs1'
  tskip = 50

  try:
    opts, args = getopt.getopt(argv, "hd:v:t:", ["help", "dir=", "var=", "tskip="])
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
    elif opt in ("-t", "--tskip"):
      tskip = int(arg)

  return dirnm, varnm, tskip

dirnm, var, tskip = getOptions(sys.argv[1:])

print(
    "<<<\n"\
    "   out dir   = %s\n"\
    "   var name  = %s\n"\
    "   time skip = %d\n"\
    ">>>\n"
    % (dirnm, var, tskip))

par = json.loads(open("params.json").read())
NX = int(par['NX'])
NY = int(par['NY'])
NZ = int(par['NZ'])
PX = int(par['PX'])
PY = int(par['PY'])
PZ = int(par['PZ'])
DT = float(par['DT'])
DH = float(par['DH'])

TMAX = float(par['TMAX'])
TIME_SKIP = int(par['EXPORT_TIME_SKIP'])
NT = int(TMAX/DT/TIME_SKIP)

ni = int(NX / PX)
nj = int(NY / PY)
nk = int(NZ / PZ)

nx = ni + 6
ny = nj + 6
nz = nk + 6

i0 = NX/2
i = int(i0 / ni)

def gather_snap(it):
  V = np.zeros([NZ, NY])
  for j in range(PY):
    for k in range(PZ):
      j1 = j * nj; j2 = j1 + nj
      k1 = k * nk; k2 = k1 + nk
      fnm = '%s/fault_mpi%02d%02d%02d.nc' % (dirnm, i, j, k)
      nc = Dataset(fnm)
      v = nc.variables[var][it,:,:]
      V[k1:k2, j1:j2] = v
  return V

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

ND = 50 + int(3e3/DH)
ND = 1
X, Y, Z = gather_coord()
X = X[ND:, ND:-ND]
Y = Y[ND:, ND:-ND]
Z = Z[ND:, ND:-ND]
cmap = 'seismic'
cmap = 'jet'
#cmap = 'hot'
#fig, ax = plt.subplots(1, 1, figsize=(18,5))
fig, ax = plt.subplots(1, 1,figsize=(6,3.5))
#fig, ax = plt.subplots(1, 1)
for it in range(0, NT, tskip):
  V = gather_snap(it)
  V = V[ND:, ND:-ND]
  #V = gaussian_filter(V, sigma=1)
  #if var == 'Vs1':
  #  V = np.log10(np.abs(V)+1e-12)
  #V = -V
  
  vm = np.max(np.abs(V))/2

  #plt.imshow(-V, vmax=vm, cmap='jet', origin='lower', interpolation='nearest')
  #plt.imshow(V, cmap=cmap, vmin=-vm, vmax=vm, origin='lower')
  #plt.imshow(-V, cmap=cmap, vmax=5, origin='lower')
  if var == 'Vs1':
    im = -V
    vmin = 0
    vmax = 5
    vmin = np.min(im)
    vmax = np.max(im)
    unit = '(m/s)'
  elif var == 'ts1':
    im = -V*1e-6
    vmin = np.min(im)
    vmax = np.max(im)
    unit = '(MPa)'
  elif var == 'State':
    im = V
    vmin = np.min(im)
    vmax = np.max(im)
    unit = ''
    #vmin = 80
    #vmax = 100
  plt.clf()
  ax.cla()
  ax = plt.gca()
  im = plt.pcolormesh(Y*1e-3, -Z*1e-3, im, cmap=cmap, vmin=vmin, vmax=vmax)
  #im = plt.imshow( -V, cmap=cmap, vmax=5, origin='lower')
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="3%", pad=0.05)
  ax.invert_yaxis()
  ax.set_aspect('equal')
  ax.set_xlabel('Along strike (km)')
  ax.set_ylabel('Downdip (km)')
  ax.set_title('%s %s @ t = %.2f sec' % (var, unit, it*DT*TIME_SKIP))
  #plt.xlabel('Along strike (km)')
  #plt.ylabel('Downdip (km)')
  #plt.title('%s, t(%d) = %.2f sec' % (var, it, it*DT*TIME_SKIP))
  #plt.title('%s (%.2f sec)' % (var, it*DT*TIME_SKIP))
  
  plt.colorbar(im, cax=cax)
  #plt.axis('image')
  #plt.colorbar(shrink=1)
  #plt.tight_layout()
  plt.pause(1e-2)
  picnm = 'snapshots/%s_it%06d.png' % (var, it)
  #print(picnm)
  #plt.savefig(picnm, bbox_inches='tight',dpi=200)
  #plt.savefig(picnm,dpi=200)
plt.show()
