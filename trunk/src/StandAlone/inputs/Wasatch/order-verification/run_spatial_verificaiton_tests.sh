#! /bin/bash
mpirun -np 4 sus inputs/Wasatch/rk3ssp-verification/TGVortex_spatial_verification_32x32.ups
mpirun -np 4 sus inputs/Wasatch/rk3ssp-verification/TGVortex_spatial_verification_64x64.ups
mpirun -np 4 sus inputs/Wasatch/rk3ssp-verification/TGVortex_spatial_verification_128x128.ups
mpirun -np 4 sus inputs/Wasatch/rk3ssp-verification/TGVortex_spatial_verification_256x256.ups
mpirun -np 4 sus inputs/Wasatch/rk3ssp-verification/TGVortex_spatial_verification_512x512.ups