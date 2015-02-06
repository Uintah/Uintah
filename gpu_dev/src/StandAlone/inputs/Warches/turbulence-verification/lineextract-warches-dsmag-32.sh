#!/bin/bash
./lineextract -pr 32 -v uVelocitySPBC -timestep 0 -istart 0 0 0 -iend 31 31 31 -o uvel_32_warches_dsmag_t0.0s.txt -uda warches-decay-isotropic-turbulence-dsmag-32.uda
./lineextract -pr 32 -v vVelocitySPBC -timestep 0 -istart 0 0 0 -iend 31 31 31 -o vvel_32_warches_dsmag_t0.0s.txt -uda warches-decay-isotropic-turbulence-dsmag-32.uda
./lineextract -pr 32 -v wVelocitySPBC -timestep 0 -istart 0 0 0 -iend 31 31 31 -o wvel_32_warches_dsmag_t0.0s.txt -uda warches-decay-isotropic-turbulence-dsmag-32.uda
#
./lineextract -pr 32 -v uVelocitySPBC -timestep 28 -istart 0 0 0 -iend 31 31 31 -o uvel_32_warches_dsmag_t0.28s.txt -uda warches-decay-isotropic-turbulence-dsmag-32.uda
./lineextract -pr 32 -v vVelocitySPBC -timestep 28 -istart 0 0 0 -iend 31 31 31 -o vvel_32_warches_dsmag_t0.28s.txt -uda warches-decay-isotropic-turbulence-dsmag-32.uda
./lineextract -pr 32 -v wVelocitySPBC -timestep 28 -istart 0 0 0 -iend 31 31 31 -o wvel_32_warches_dsmag_t0.28s.txt -uda warches-decay-isotropic-turbulence-dsmag-32.uda
#
./lineextract -pr 32 -v uVelocitySPBC -timestep 66 -istart 0 0 0 -iend 31 31 31 -o uvel_32_warches_dsmag_t0.66s.txt -uda warches-decay-isotropic-turbulence-dsmag-32.uda
./lineextract -pr 32 -v vVelocitySPBC -timestep 66 -istart 0 0 0 -iend 31 31 31 -o vvel_32_warches_dsmag_t0.66s.txt -uda warches-decay-isotropic-turbulence-dsmag-32.uda
./lineextract -pr 32 -v wVelocitySPBC -timestep 66 -istart 0 0 0 -iend 31 31 31 -o wvel_32_warches_dsmag_t0.66s.txt -uda warches-decay-isotropic-turbulence-dsmag-32.uda
