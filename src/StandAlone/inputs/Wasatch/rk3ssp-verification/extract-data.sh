#! /bin/bash
./lineextract -v XSVOL -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 1 -o XSVOL_512.txt -uda projection_rk3_verification_dt0.01s.uda
./lineextract -v YSVOL -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 1 -o YSVOL_512.txt -uda projection_rk3_verification_dt0.01s.uda

./lineextract -v XXVOL -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 1 -o XXVOL_512.txt -uda projection_rk3_verification_dt0.01s.uda
./lineextract -v XYVOL -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 1 -o XYVOL_512.txt -uda projection_rk3_verification_dt0.01s.uda

./lineextract -v YXVOL -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 1 -o YXVOL_512.txt -uda projection_rk3_verification_dt0.01s.uda
./lineextract -v YYVOL -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 1 -o YYVOL_512.txt -uda projection_rk3_verification_dt0.01s.uda

./lineextract -v x-mom -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 1 -o xmom_dt_0_01_timestep_1.txt -uda projection_rk3_verification_dt0.01s.uda
./lineextract -v y-mom -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 1 -o ymom_dt_0_01_timestep_1.txt -uda projection_rk3_verification_dt0.01s.uda
./lineextract -v pressure -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 1 -o p_dt_0_01_timestep_1.txt -uda projection_rk3_verification_dt0.01s.uda

./lineextract -v x-mom -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 2 -o xmom_dt_0_005_timestep_2.txt -uda projection_rk3_verification_dt0.005s.uda
./lineextract -v y-mom -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 2 -o ymom_dt_0_005_timestep_2.txt -uda projection_rk3_verification_dt0.005s.uda
./lineextract -v pressure -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 2 -o p_dt_0_005_timestep_2.txt -uda projection_rk3_verification_dt0.005s.uda

./lineextract -v x-mom -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 2 -o xmom_dt_0_005_timestep_2.txt -uda projection_rk3_verification_dt0.005s.uda
./lineextract -v y-mom -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 2 -o ymom_dt_0_005_timestep_2.txt -uda projection_rk3_verification_dt0.005s.uda
./lineextract -v pressure -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 2 -o p_dt_0_005_timestep_2.txt -uda projection_rk3_verification_dt0.005s.uda

./lineextract -v x-mom -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 4 -o xmom_dt_0_0025_timestep_4.txt -uda projection_rk3_verification_dt0.0025s.uda
./lineextract -v y-mom -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 4 -o ymom_dt_0_0025_timestep_4.txt -uda projection_rk3_verification_dt0.0025s.uda
./lineextract -v pressure -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 4 -o p_dt_0_0025_timestep_4.txt -uda projection_rk3_verification_dt0.0025s.uda

./lineextract -v x-mom -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 8 -o xmom_dt_0_00125_timestep_8.txt -uda projection_rk3_verification_dt0.00125s.uda
./lineextract -v y-mom -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 8 -o ymom_dt_0_00125_timestep_8.txt -uda projection_rk3_verification_dt0.00125s.uda
./lineextract -v pressure -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 8 -o p_dt_0_00125_timestep_8.txt -uda projection_rk3_verification_dt0.00125s.uda

./lineextract -v x-mom -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 16 -o xmom_dt_0_000625_timestep_16.txt -uda projection_rk3_verification_dt0.000625s.uda
./lineextract -v y-mom -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 16 -o ymom_dt_0_000625_timestep_16.txt -uda projection_rk3_verification_dt0.000625s.uda
./lineextract -v pressure -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 16 -o p_dt_0_000625_timestep_16.txt -uda projection_rk3_verification_dt0.000625s.uda