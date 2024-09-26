#!/usr/bin/env bash


gmt begin tpv10_rupture_front
  gmt set FONT_ANNOT_PRIMARY = 12p,4,black
  gmt set FONT_LABEL = 14p,4,black
  gmt basemap -R-15/15/0/15 -JX10c/-5c -BWS -B5 \
      -Bx+l"Along strike direction (km)" \
      -By+l"Down-dip direction (km)"
  gmt basemap -R-15/15/0/15 -JX10c/-5c -Ben
  gmt grdcontour @tpv10_init_t0_barall.nc -C0.5 -W2.0p,gray
  gmt grdcontour @tpv10_init_t0_fortran.nc -C0.5 -W1.5p,blue,--
  gmt grdcontour @tpv10_init_t0_gpu.nc -C0.5 -W0.5p,red
  echo "0 12
  0 7.5
  0 4.5
  4.5 7.5
  "|gmt psxy -St0.4c -Gblack

gmt end
