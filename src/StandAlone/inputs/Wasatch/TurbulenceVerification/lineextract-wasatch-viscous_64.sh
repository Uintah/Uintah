#!/bin/bash
./lineextract -pr 64 -v x-mom -timestep 0 -istart 0 0 0 -iend 63 63 63 -o uvel_64_wasatch_viscous_t0.0s.txt -uda decay-isotropic-turbulence-viscous-64.uda
./lineextract -pr 64 -v y-mom -timestep 0 -istart 0 0 0 -iend 63 63 63 -o vvel_64_wasatch_viscous_t0.0s.txt -uda decay-isotropic-turbulence-viscous-64.uda
./lineextract -pr 64 -v z-mom -timestep 0 -istart 0 0 0 -iend 63 63 63 -o wvel_64_wasatch_viscous_t0.0s.txt -uda decay-isotropic-turbulence-viscous-64.uda
#
./lineextract -pr 64 -v x-mom -timestep 28 -istart 0 0 0 -iend 63 63 63 -o uvel_64_wasatch_viscous_t0.28s.txt -uda decay-isotropic-turbulence-viscous-64.uda
./lineextract -pr 64 -v y-mom -timestep 28 -istart 0 0 0 -iend 63 63 63 -o vvel_64_wasatch_viscous_t0.28s.txt -uda decay-isotropic-turbulence-viscous-64.uda
./lineextract -pr 64 -v z-mom -timestep 28 -istart 0 0 0 -iend 63 63 63 -o wvel_64_wasatch_viscous_t0.28s.txt -uda decay-isotropic-turbulence-viscous-64.uda
#
./lineextract -pr 64 -v x-mom -timestep 66 -istart 0 0 0 -iend 63 63 63 -o uvel_64_wasatch_viscous_t0.66s.txt -uda decay-isotropic-turbulence-viscous-64.uda
./lineextract -pr 64 -v y-mom -timestep 66 -istart 0 0 0 -iend 63 63 63 -o vvel_64_wasatch_viscous_t0.66s.txt -uda decay-isotropic-turbulence-viscous-64.uda
./lineextract -pr 64 -v z-mom -timestep 66 -istart 0 0 0 -iend 63 63 63 -o wvel_64_wasatch_viscous_t0.66s.txt -uda decay-isotropic-turbulence-viscous-64.uda
