#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys
import json

plt.rcParams['font.size'] = 14

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
      '         -l             : log scale\n'+\
      'Example:\n'+\
      '         ./draw_fault_seismo.py\n'+\
      '         ./draw_fault_seismo.py -v Vs1\n'+\
      '         ./draw_fault_seismo.py -d ouput -v Vs1\n'+\
      '         ./draw_fault_seismo.py -d ouput -v Vs1 -j 200 -k 125\n'+\
      '         ./draw_fault_seismo.py -d ouput -v Vs1 -j 200 -k 125 -r\n'+\
      '         ./draw_fault_seismo.py -d ouput -v Vs1 -j 200 -k 125 -l\n'+\
      '')

def getOptions(argv):
  import getopt
  dirnm = 'output'
  varnm = 'ts1'
  j = 10
  k = 10
  flag_reverse = 0
  flag_log = 0

  try:
    opts, args = getopt.getopt(argv, "hd:v:j:k:rl", ["help", "dir=", "var=", "--istrke", "--idip=", "reverse", "log"])
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
      j = float(arg)
    elif opt in ("-k", "--idip"):
      k = float(arg)
    elif opt in ("-r", "--reverse"):
      flag_reverse = 1
    elif opt in ("-l", "--log"):
      flag_log = 1

  return dirnm, varnm, j, k, flag_reverse, flag_log

dirnm, var, j, k, flag_reverse, flag_log = getOptions(sys.argv[1:])


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
jkm = j
kkm = k
j = int(j*1e3/DH) + int(NY/2)
k = NZ - 1  - int(k*1e3/DH)

dimx = int(i / ni)
dimy = int(j / nj)
dimz = int(k / nk)
local_j = j - dimy * nj
local_k = k - dimz * nk

print(
    "<<<\n"\
    "   out dir   = %s\n"\
    "   var name  = %s\n"\
    "   index j   = %d\n"\
    "   index k   = %d\n"\
    "   reverse   = %d\n"\
    "   log       = %d\n"\
    ">>>\n"
    % (dirnm, var, j, k, flag_reverse, flag_log))
#print("local( j = %d, k = %d)" % (local_j, local_k))
fnm = './%s/fault_mpi%02d%02d%02d.nc' % (dirnm, dimx, dimy, dimz)
#print(fnm)

nc = Dataset(fnm)
v = nc.variables[var][::1,local_k,local_j]

NT = len(v)

t = np.arange(0, NT, 1) * DT * Tskip

#v = np.abs(v) + 1e-12
#v = np.log10(v)
#
if(var == 'State'):
  RS_f0 = 0.6
  RS_b = 0.012
  RS_L = 0.02
  RS_V0 = 1e-6
  psi = v
  #v = (psi - RS_f0)/RS_b + np.log(RS_L / RS_V0)
  v = RS_L / RS_V0 * np.exp((psi - RS_f0)/RS_b)
  v = np.log10(v)

#v = np.log10(np.exp(v))
fnm = './ma2/strike%.1fdip%.1f' % (jkm, kkm)
print(fnm)

v0 = np.loadtxt(fnm, comments='#')
if var == 'ts1':
  v0 = v0[:, 3]
  v = v * 1e-6
elif var == 'State':
  v0 = v0[:, 8]
else:
  v0 = v0[:, 2]

t0 = np.arange(len(v0)) * 0.008

if flag_reverse:
  v = -v

if flag_log:
  v = np.log10(np.abs(v)+1e-12)

#plt.figure(figsize=(8,4))
fig, ax = plt.subplots(1, 1, figsize=(8,4.5))
#fig.set_size_inches(7, 4)
ax.plot(t0, v0, color='b', label='FEM')
ax.plot(t, v, color='r',linestyle='-', label='CG-FDM')
#plt.title('%s (strike %.1f km; dip %.1f km)' % (var,y/1e3,z/1e3))
plt.title('strike = %g km, dip = %g km' % ( jkm, kkm),fontsize=14)
plt.xlabel('Time (sec)')
if var == 'ts1':
  plt.ylabel('Stress (MPa)')
if var == 'Vs1':
  plt.ylabel('Velocity (m/s)')
if var == 'State':
  plt.ylabel('log10 theta')

ax.legend(frameon=False)
#filenm = '%s_strike%s_dip%s.png' % (sys.argv[1], sys.argv[2], sys.argv[3])
#fig.tight_layout(#)
#picnm = './seis/strike%.1fdip%.1f_%s_dh%d.png' % (jkm, kkm, var, DH)
#plt.savefig(picnm,bbox_inches='tight',dpi=300)
#picnm = './seis/strike%.1fdip%.1f_%s_dh%d.pdf' % (jkm, kkm, var, DH)
#plt.savefig(picnm,bbox_inches='tight')
plt.show()
