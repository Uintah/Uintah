#!/bin/sh

# Not supper pretty but it works.  -Jeremy
# 
#  Instructions:

#  Before you start make sure that you have a compiled version of 
#  the code with VERIFY_TIMEINT defined in ExplicitTimeInt.h.  This 
#  will activate the appropriate code for verification purposes. 

#  1) Copy verify_time_int.ups into your working directory
#  2) Copy this script into your working directory
#  3) On an interactive node, execute this script
#  4) Use the Matlab script (plot_time_error.m) to plot the order
#     of convergence. Note that the error is saved in the *.txt 
#     from this script.  

usage()
{
  echo ""
  echo "Usage: $0 <num of equations> <path to sus>"
  echo ""
  exit 
}

if test $# = 0; then 
  echo ""
  echo "Bad number of arguments..."
  usage
  exit 
fi

if test $1 = "-h" || test $1 = "-help" || test $1 = "--help"; then 
  usage 
  exit
fi

# I think this script only works on 1 proc. Check this later...
NP=1

echo "Performing time integration verification using the ExplicitTimeInt class"
echo " Begining "

echo "-------------------------------------------------------------------------"
echo " Running Forward Euler verification "
echo "-------------------------------------------------------------------------"

## Grid 1
mpirun -m $PBS_NODEFILE -np NP $2 -mpi verify_time_int.ups >& FE_output_G1.out
sed "s/<timestep_multiplier>1/<timestep_multiplier>.5/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups
sed "s/<max_Timesteps>2/<max_Timesteps>4/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups

##Grid 2
mpirun -m $PBS_NODEFILE -np NP $2 -mpi verify_time_int.ups >& FE_output_G2.out
sed "s/<timestep_multiplier>.5/<timestep_multiplier>.25/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups
sed "s/<max_Timesteps>4/<max_Timesteps>8/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups

##Grid 3
mpirun -m $PBS_NODEFILE -np NP $2 -mpi verify_time_int.ups >& FE_output_G3.out
sed "s/<timestep_multiplier>.25/<timestep_multiplier>.125/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups
sed "s/<max_Timesteps>8/<max_Timesteps>16/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups

##Grid 4
mpirun -m $PBS_NODEFILE -np NP $2 -mpi verify_time_int.ups >& FE_output_G4.out

echo "-------------------------------------------------------------------------"
echo " Running SSP-RK2 verification "
echo "-------------------------------------------------------------------------"
sed "s/first/second/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups
sed "s/FE/RK2SSP/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups

sed "s/<timestep_multiplier>.125/<timestep_multiplier>1/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups
sed "s/<max_Timesteps>16/<max_Timesteps>2/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups

## Grid 1
mpirun -m $PBS_NODEFILE -np NP $2 -mpi verify_time_int.ups >& RK2_output_G1.out
sed "s/<timestep_multiplier>1/<timestep_multiplier>.5/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups
sed "s/<max_Timesteps>2/<max_Timesteps>4/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups

##Grid 2
mpirun -m $PBS_NODEFILE -np NP $2 -mpi verify_time_int.ups >& RK2_output_G2.out
sed "s/<timestep_multiplier>.5/<timestep_multiplier>.25/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups
sed "s/<max_Timesteps>4/<max_Timesteps>8/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups

##Grid 3
mpirun -m $PBS_NODEFILE -np NP $2 -mpi verify_time_int.ups >& RK2_output_G3.out
sed "s/<timestep_multiplier>.25/<timestep_multiplier>.125/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups
sed "s/<max_Timesteps>8/<max_Timesteps>16/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups

##Grid 4
mpirun -m $PBS_NODEFILE -np NP $2 -mpi verify_time_int.ups >& RK2_output_G4.out

echo "-------------------------------------------------------------------------"
echo " Running SSP-RK3 verification "
echo "-------------------------------------------------------------------------"
sed "s/second/third/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups
sed "s/RK2SSP/RK3SSP/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups

sed "s/<timestep_multiplier>.125/<timestep_multiplier>1/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups
sed "s/<max_Timesteps>16/<max_Timesteps>2/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups

## Grid 1
mpirun -m $PBS_NODEFILE -np NP $2 -mpi verify_time_int.ups >& RK3_output_G1.out
sed "s/<timestep_multiplier>1/<timestep_multiplier>.5/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups
sed "s/<max_Timesteps>2/<max_Timesteps>4/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups

##Grid 2
mpirun -m $PBS_NODEFILE -np NP $2 -mpi verify_time_int.ups >& RK3_output_G2.out
sed "s/<timestep_multiplier>.5/<timestep_multiplier>.25/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups
sed "s/<max_Timesteps>4/<max_Timesteps>8/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups

##Grid 3
mpirun -m $PBS_NODEFILE -np NP $2 -mpi verify_time_int.ups >& RK3_output_G3.out
sed "s/<timestep_multiplier>.25/<timestep_multiplier>.125/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups
sed "s/<max_Timesteps>8/<max_Timesteps>16/" verify_time_int.ups > newfile
mv newfile verify_time_int.ups

##Grid 4
mpirun -m $PBS_NODEFILE -np NP $2 -mpi verify_time_int.ups >& RK3_output_G4.out

#-- now extract the useful information from the files:
grep "Error from time integration" FE_output_G1.out > newfile
sed "s/Error from time integration =//" newfile > FE_G1.txt
tail -n $1 FE_G1.txt > newfile
mv newfile FE_G1.txt
grep "Error from time integration" FE_output_G2.out > newfile
sed "s/Error from time integration =//" newfile > FE_G2.txt
tail -n $1 FE_G2.txt > newfile
mv newfile FE_G2.txt
grep "Error from time integration" FE_output_G3.out > newfile
sed "s/Error from time integration =//" newfile > FE_G3.txt
tail -n $1 FE_G3.txt > newfile
mv newfile FE_G3.txt
grep "Error from time integration" FE_output_G4.out > newfile
sed "s/Error from time integration =//" newfile > FE_G4.txt
tail -n $1 FE_G4.txt > newfile
mv newfile FE_G4.txt

grep "Error from time integration" RK2_output_G1.out > newfile
sed "s/Error from time integration =//" newfile > RK2_G1.txt
tail -n $1 RK2_G1.txt > newfile
mv newfile RK2_G1.txt
grep "Error from time integration" RK2_output_G2.out > newfile
sed "s/Error from time integration =//" newfile > RK2_G2.txt
tail -n $1 RK2_G2.txt > newfile
mv newfile RK2_G2.txt
grep "Error from time integration" RK2_output_G3.out > newfile
sed "s/Error from time integration =//" newfile > RK2_G3.txt
tail -n $1 RK2_G3.txt > newfile
mv newfile RK2_G3.txt
grep "Error from time integration" RK2_output_G4.out > newfile
sed "s/Error from time integration =//" newfile > RK2_G4.txt
tail -n $1 RK2_G4.txt > newfile
mv newfile RK2_G4.txt

grep "Error from time integration" RK3_output_G1.out > newfile
sed "s/Error from time integration =//" newfile > RK3_G1.txt
tail -n $1 RK3_G1.txt > newfile
mv newfile RK3_G1.txt
grep "Error from time integration" RK3_output_G2.out > newfile
sed "s/Error from time integration =//" newfile > RK3_G2.txt
tail -n $1 RK3_G2.txt > newfile
mv newfile RK3_G2.txt
grep "Error from time integration" RK3_output_G3.out > newfile
sed "s/Error from time integration =//" newfile > RK3_G3.txt
tail -n $1 RK3_G3.txt > newfile
mv newfile RK3_G3.txt
grep "Error from time integration" RK3_output_G4.out > newfile
sed "s/Error from time integration =//" newfile > RK3_G4.txt
tail -n $1 RK3_G4.txt > newfile
mv newfile RK3_G4.txt

##--- clean up 
rm -rf time_verification.uda*
