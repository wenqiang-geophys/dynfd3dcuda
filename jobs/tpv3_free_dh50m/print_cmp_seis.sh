#!/bin/bash

jvec=(0.0 0.0  0.0 9.0 12.0 12.0)
kvec=(7.5 3.0 12.0 7.5  3.0 12.0)

for((i=0;i<${#jvec[*]};i++))
do
echo ${jvec[$i]} ${kvec[$i]}
./cmp_seis.py -j ${jvec[$i]} -k ${kvec[$i]} -v Vs1 -r &
./cmp_seis.py -j ${jvec[$i]} -k ${kvec[$i]} -v ts1 -r &
./cmp_seis.py -j ${jvec[$i]} -k ${kvec[$i]} -v State  &
done
