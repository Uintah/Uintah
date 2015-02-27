echo
echo "-------------------------------------------------------------------------------"
echo " Starting a simulation with the fastest run configuration for 25 sampling rays"
echo "-------------------------------------------------------------------------------"
echo

mpirun -genv SCI_DEBUG="ComponentTimings:+,ExecTimes:+,ThreadedMPI_CompactAffinity:-" -genv I_MPI_MIC=enable -np 1 ./sus.mic-example -mpi -nthreads 242 25r480p.ups | tee 25-ray-results.txt


mv exectimes.1.0 exectimes.25-ray-results
date

echo
echo "-------------------------------------------------------------------------------"
echo " Starting a simulation with the fastest run configuration for 50 sampling rays"
echo "-------------------------------------------------------------------------------"
echo

mpirun -genv SCI_DEBUG="ComponentTimings:+,ExecTimes:+,ThreadedMPI_CompactAffinity:-" -genv I_MPI_MIC=enable -np 1 ./sus.mic-example -mpi -nthreads 242 50r480p.ups | tee 50-ray-results.txt

mv exectimes.1.0 exectimes.50-ray-results
date

echo
echo "-------------------------------------------------------------------------------"
echo " Starting a simulation with the fastest run configuration for 100 sampling rays"
echo "-------------------------------------------------------------------------------"
echo

mpirun -genv SCI_DEBUG="ComponentTimings:+,ExecTimes:+,ThreadedMPI_CompactAffinity:-" -genv I_MPI_MIC=enable -np 1 ./sus.mic-example -mpi -nthreads 244 100r486p.ups | tee 100-ray-results.txt

mv exectimes.1.0 exectimes.100-ray-results
date

echo
echo "Going down successfully"
