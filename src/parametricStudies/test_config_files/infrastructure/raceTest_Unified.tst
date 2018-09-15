<?xml version="1.0" encoding="ISO-8859-1"?>
<start>

 <!-- optional path to inputs directory 
      default is src/StandAlone/inputs  -->
<inputs_path></inputs_path>


<upsFile>ICE/advect.ups</upsFile>

<AllTests>
  <replace_lines>
    <max_Timesteps>10           </max_Timesteps>
    <resolution> [100,100,100]  </resolution>
    <patches>      [3,3,3]      </patches>
  </replace_lines>
</AllTests>

<Test>
    <Title>1_2</Title>
    <sus_cmd>mpirun -np 1 sus -mpi -nthreads 2</sus_cmd>
</Test>

<Test>
    <Title>1_4</Title>
    <sus_cmd>mpirun -np 1 sus -mpi -nthreads 4</sus_cmd>
</Test>
<Test>
    <Title>1_8</Title>
    <sus_cmd>mpirun -np 1 sus -mpi -nthreads 8</sus_cmd>
</Test>
<Test>
    <Title>1_16</Title>
    <sus_cmd>mpirun -np 1 sus -mpi -nthreads 16</sus_cmd>
</Test>
<Test>
    <Title>1_32</Title>
    <sus_cmd>mpirun -np 1 sus -mpi -nthreads 32</sus_cmd>
</Test>
<Test>
    <Title>1_64</Title>
    <sus_cmd>mpirun -np 1 sus -mpi -nthreads 64</sus_cmd>
</Test>
<!--__________________________________-->

<Test>
    <Title>2_2</Title>
    <sus_cmd>mpirun -np 2 sus -mpi -nthreads 2</sus_cmd>
</Test>

<Test>
    <Title>2_4</Title>
    <sus_cmd>mpirun -np 2 sus -mpi -nthreads 4</sus_cmd>
</Test>
<Test>
    <Title>2_8</Title>
    <sus_cmd>mpirun -np 2 sus -mpi -nthreads 8</sus_cmd>
</Test>
<Test>
    <Title>2_16</Title>
    <sus_cmd>mpirun -np 2 sus -mpi -nthreads 16</sus_cmd>
</Test>
<Test>
    <Title>2_32</Title>
    <sus_cmd>mpirun -np 2 sus -mpi -nthreads 32</sus_cmd>
</Test>
<Test>
    <Title>2_64</Title>
    <sus_cmd>mpirun -np 2 sus -mpi -nthreads 64</sus_cmd>
</Test>

<!--__________________________________-->

<Test>
    <Title>4_2</Title>
    <sus_cmd>mpirun -np 4 sus -mpi -nthreads 2</sus_cmd>
</Test>

<Test>
    <Title>4_4</Title>
    <sus_cmd>mpirun -np 4 sus -mpi -nthreads 4</sus_cmd>
</Test>
<Test>
    <Title>4_8</Title>
    <sus_cmd>mpirun -np 4 sus -mpi -nthreads 8</sus_cmd>
</Test>
<Test>
    <Title>4_16</Title>
    <sus_cmd>mpirun -np 4 sus -mpi -nthreads 16</sus_cmd>
</Test>
<Test>
    <Title>4_32</Title>
    <sus_cmd>mpirun -np 4 sus -mpi -nthreads 32</sus_cmd>
</Test>
<Test>
    <Title>4_64</Title>
    <sus_cmd>mpirun -np 4 sus -mpi -nthreads 64</sus_cmd>
</Test>

</start>
