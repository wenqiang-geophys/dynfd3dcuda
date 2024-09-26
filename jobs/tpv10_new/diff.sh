#!/bin/bash

for((i=0;i<1;i++));do
  for((j=0;j<4;j++));do
    for((k=0;k<2;k++));do

      fnm1=`printf "output/fault_mpi%02d%02d%02d.nc" $i $j $k`
      fnm2=`printf "output1/fault_mpi%02d%02d%02d.nc" $i $j $k`

      echo $fnm1 $fnm2
      diff $fnm1 $fnm2

    done
  done
done
