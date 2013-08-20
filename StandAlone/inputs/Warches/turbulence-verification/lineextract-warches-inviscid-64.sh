#!/bin/bash
./lineextract -pr 64 -v uVelocitySPBC -timestep 0 -istart 0 0 0 -iend 63 63 63 -o uvel_64_warches_inviscid_t0.0s.txt -uda warches-decay-isotropic-turbulence-inviscid-64.uda
./lineextract -pr 64 -v vVelocitySPBC -timestep 0 -istart 0 0 0 -iend 63 63 63 -o vvel_64_warches_inviscid_t0.0s.txt -uda warches-decay-isotropic-turbulence-inviscid-64.uda
./lineextract -pr 64 -v wVelocitySPBC -timestep 0 -istart 0 0 0 -iend 63 63 63 -o wvel_64_warches_inviscid_t0.0s.txt -uda warches-decay-isotropic-turbulence-inviscid-64.uda
#
./lineextract -pr 64 -v uVelocitySPBC -timestep 28 -istart 0 0 0 -iend 63 63 63 -o uvel_64_warches_inviscid_t0.28s.txt -uda warches-decay-isotropic-turbulence-inviscid-64.uda
./lineextract -pr 64 -v vVelocitySPBC -timestep 28 -istart 0 0 0 -iend 63 63 63 -o vvel_64_warches_inviscid_t0.28s.txt -uda warches-decay-isotropic-turbulence-inviscid-64.uda
./lineextract -pr 64 -v wVelocitySPBC -timestep 28 -istart 0 0 0 -iend 63 63 63 -o wvel_64_warches_inviscid_t0.28s.txt -uda warches-decay-isotropic-turbulence-inviscid-64.uda
#
./lineextract -pr 64 -v uVelocitySPBC -timestep 66 -istart 0 0 0 -iend 63 63 63 -o uvel_64_warches_inviscid_t0.66s.txt -uda warches-decay-isotropic-turbulence-inviscid-64.uda
./lineextract -pr 64 -v vVelocitySPBC -timestep 66 -istart 0 0 0 -iend 63 63 63 -o vvel_64_warches_inviscid_t0.66s.txt -uda warches-decay-isotropic-turbulence-inviscid-64.uda
./lineextract -pr 64 -v wVelocitySPBC -timestep 66 -istart 0 0 0 -iend 63 63 63 -o wvel_64_warches_inviscid_t0.66s.txt -uda warches-decay-isotropic-turbulence-inviscid-64.uda
