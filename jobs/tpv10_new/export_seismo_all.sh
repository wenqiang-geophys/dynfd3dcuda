#!/bin/bash

./export_seismo.py Vs2 0 12    $1
./export_seismo.py Vs2 0 4.5   $1
./export_seismo.py Vs2 0 7.5   $1
./export_seismo.py Vs2 4.5 7.5 $1
./export_seismo.py Vs2 12 7.5  $1
