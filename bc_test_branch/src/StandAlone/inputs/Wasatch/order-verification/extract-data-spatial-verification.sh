#! /bin/bash
declare -a varnames=(x-mom y-mom x-mom_convFlux_x x-mom_convFlux_y y-mom_convFlux_x y-mom_convFlux_y tau_xx tau_yy x-mom_rhs_partial y-mom_rhs_partial x-mom_rhs_full 
y-mom_rhs_full XSVOL YSVOL XXVOL YXVOL XYVOL YYVOL)
udanames=(32x32 64x64 128x128 256x256 512x512)
dx="_dx_"
tstep="_timestep_1.txt"
udabase="TGVortex_spatial_verification_"
udaext=".uda"
baseudaname=
outnames=(xmom ymom xconvx xconvy yconvx yconvy tauxx tauyy xmomrhspart ymomrhspart xmomrhsfull ymomrhsfull XSVOL YSVOL XXVOL YXVOL XYVOL YYVOL)
echo ${outnames[1]}
for name in ${udanames[@]}
do
  i=0
  for varname in ${varnames[@]}
  do
      ./lineextract -v $varname  -istart 0 0 0 -iend 512 512 0 -pr 16 -timestep 1 -o ${outnames[i]}$dx$name$tstep -uda $udabase$name$udaext
      ((i++))
  done
done