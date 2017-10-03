#! /bin/bash
mpirun -np 4 sus inputs/Wasatch/rk3ssp-verification/projection_rk3_verification_dt0.01s.ups
mpirun -np 4 sus inputs/Wasatch/rk3ssp-verification/projection_rk3_verification_dt0.005s.ups
mpirun -np 4 sus inputs/Wasatch/rk3ssp-verification/projection_rk3_verification_dt0.0025s.ups
mpirun -np 4 sus inputs/Wasatch/rk3ssp-verification/projection_rk3_verification_dt0.00125s.ups
mpirun -np 4 sus inputs/Wasatch/rk3ssp-verification/projection_rk3_verification_dt0.000625s.ups