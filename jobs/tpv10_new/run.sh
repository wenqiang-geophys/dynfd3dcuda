#!/usr/bin/bash

#set -x
set -e

#python config_params.py

PX=`python get_params.py PX`
PY=`python get_params.py PY`
PZ=`python get_params.py PZ`
OUT=`python get_params.py OUT`

#NP=`echo "$PX*$PY*$PZ"|bc -q`
NP=`echo $PX $PY $PZ|awk '{print $1*$2*$3}'`
echo NP=$NP

echo "garray3 slots=4
garray4 slots=4" > nodelists

EXE="../../bin/a.out"

#rm -rf $OUT && mkdir -p $OUT
mkdir -p $OUT

RUN=/home/wqzhang/install/openmpi-gnu/bin/mpirun

if [ $NP -eq 1 ];then
echo "serial"
${EXE}
else
$RUN -np $NP -x LD_LIBRARY_PATH --machinefile nodelists ${EXE} 2>&1|tee log
fi
