echo
echo "-------------------------------------------------------"
echo " Starting a simulation with 2 threads per physical core"
echo "-------------------------------------------------------"
echo

mpirun -genv SCI_DEBUG="ComponentTimings:+,ExecTimes:+,ThreadedMPI_SelectiveAffinity:+" -genv I_MPI_MIC=enable -np 1 ./sus.mic-example -mpi -nthreads 121 100r480p.ups | tee 2tppc-results.txt

mv exectimes.1.0 exectimes.2tppc-results
date

echo
echo "-------------------------------------------------------"
echo " Starting a simulation with 3 threads per physical core"
echo "-------------------------------------------------------"
echo

mpirun -genv SCI_DEBUG="ComponentTimings:+,ExecTimes:+,ThreadedMPI_ScatterAffinity:+" -genv I_MPI_MIC=enable -np 1 ./sus.mic-example -mpi -nthreads 181 100r720p.ups | tee 3tppc-results.txt

mv exectimes.1.0 exectimes.3tppc-results
date

echo
echo "-------------------------------------------------------"
echo " Starting a simulation with 4 threads per physical core"
echo "-------------------------------------------------------"
echo

mpirun -genv SCI_DEBUG="ComponentTimings:+,ExecTimes:+,ThreadedMPI_CompactAffinity:-" -genv I_MPI_MIC=enable -np 1 ./sus.mic-example -mpi -nthreads 244 100r486p.ups | tee 4tppc-results.txt

mv exectimes.1.0 exectimes.4tppc-results
date

echo
echo "Going down successfully"
