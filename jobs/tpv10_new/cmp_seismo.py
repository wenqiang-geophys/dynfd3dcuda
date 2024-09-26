#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys
import json

def usage():
  print(
      '  Usage: ./draw_fault_seismo.py -d <directory>\n'+\
      '                                -v <variable>\n'+\
      '                                -j <index_strike>\n'+\
      '                                -k <index_dip>\n'+\
      '                                -r\n'+\
      '     or: ./draw_fault_seismo.py --dir <directory>\n'+\
      '                                --var <variable>\n'+\
      '                                --istrike <index_strike>\n'+\
      '                                --idip <index_dip>\n'+\
      '                                --reverse\n'+\
      'Default:\n'+\
      '         <directory>    : output\n'+\
      '         <variable>     : ts1\n'+\
      '         <index_strike> : 10\n'+\
      '         <index_dip>    : 10\n'+\
      '         -r             : reverse seismo\n'+\
      'Example:\n'+\
      '         ./draw_fault_seismo.py\n'+\
      '         ./draw_fault_seismo.py -v Vs1\n'+\
      '         ./draw_fault_seismo.py -d ouput -v Vs1\n'+\
      '         ./draw_fault_seismo.py -d ouput -v Vs1 -j 200 -k 125\n'+\
      '         ./draw_fault_seismo.py -d ouput -v Vs1 -j 200 -k 125 -r\n'+\
      '')

def getOptions(argv):
  import getopt
  dirnm = 'output'
  varnm = 'ts1'
  j = 10
  k = 10
  flag_reverse = 0

  try:
    opts, args = getopt.getopt(argv, "hd:v:j:k:r", ["help", "dir=", "var=", "--istrke", "--idip=", "reverse"])
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
    elif opt in ("-j", "--istrike"):
      j = int(arg)
    elif opt in ("-k", "--idip"):
      k = int(arg)
    elif opt in ("-r", "--reverse"):
      flag_reverse = 1

  return dirnm, varnm, j, k, flag_reverse

dirnm, var, j, k, flag_reverse = getOptions(sys.argv[1:])

print(
    "<<<\n"\
    "   out dir   = %s\n"\
    "   var name  = %s\n"\
    "   index j   = %d\n"\
    "   index k   = %d\n"\
    "   reverse   = %d\n"\
    ">>>\n"
    % (dirnm, var, j, k, flag_reverse))

par = json.loads(open("params.json").read())
NX = int(par['NX'])
NY = int(par['NY'])
NZ = int(par['NZ'])
PX = int(par['PX'])
PY = int(par['PY'])
PZ = int(par['PZ'])

DH = float(par['DH'])
TMAX = float(par['TMAX'])
Tskip = float(par['EXPORT_TIME_SKIP'])
DT = float(par['DT'])

ni = int(NX / PX)
nj = int(NY / PY)
nk = int(NZ / PZ)

i = int(NX / 2)

dimx = int(i / ni)
dimy = int(j / nj)
dimz = int(k / nk)
local_j = j - dimy * nj
local_k = k - dimz * nk

#print("local( j = %d, k = %d)" % (local_j, local_k))
fnm = './%s/fault_mpi%02d%02d%02d.nc' % (dirnm, dimx, dimy, dimz)
#print(fnm)

nc = Dataset(fnm)
v = nc.variables[var][::1,local_k,local_j]

NT = len(v)

t = np.arange(0, NT, 1) * DT * Tskip

fnm = './%s/fault_mpi%02d%02d%02d.nc' % ('output1', dimx, dimy, dimz)

nc = Dataset(fnm)
v1 = nc.variables[var][::1,local_k,local_j]
NT = len(v1)

t1 = np.arange(0, NT, 1) * DT * Tskip
#v = np.abs(v) + 1e-12
#v = np.log10(v)
#
#if(var == 'State'):
#  RS_f0 = 0.6
#  RS_b = 0.012
#  RS_L = 0.02
#  RS_V0 = 1e-6
#  psi = v
#  v = (psi - RS_f0)/RS_b + np.log(RS_L / RS_V0)
#  v = np.log10(np.exp(v))

if flag_reverse:
  v = -v
  v1 = -v1

#plt.figure(figsize=(8,4))
fig, ax = plt.subplots(1, 1)
#fig.set_size_inches(7, 4)
plt.plot(t, v)
plt.plot(t1, v1)
#plt.title('%s (strike %.1f km; dip %.1f km)' % (var,y/1e3,z/1e3))
plt.title('%s (j = %d, k = %d)' % (var, j, k))
plt.xlabel('Time (sec)')
#filenm = '%s_strike%s_dip%s.png' % (sys.argv[1], sys.argv[2], sys.argv[3])
#fig.tight_layout()
#plt.savefig(filenm)
plt.show()
