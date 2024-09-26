#!/bin/bash

var=State
convert snapshots/${var}_it*.png -loop 0 -delay 100 ${var}.gif
