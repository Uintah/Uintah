<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>RMCRT_bm1_1L.ups</upsFile>
<gnuplot>
  <script>plotM-patches.gp</script>
  <title>GPU:RMCRT Error vs Patch configuration \\n 1 Timestep, 64^3, 512 Rays per Cell</title>
  <ylabel>Error (L2 norm)</ylabel>
  <xlabel>Different Patch configurations (x,y,z)</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
    <max_Timesteps>1</max_Timesteps>
    <randomSeed> true </randomSeed>
    <resolution> [64,64,64]  </resolution>
    <nDivQRays>   512        </nDivQRays>
  </replace_lines>
</AllTests>

<Test>
    <Title>1_1_1</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 2 -gpu </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>1_1_1</x>
    <replace_lines>
      <patches>  [1,1,1]   </patches>
    </replace_lines>
</Test>
<Test>
    <Title>2_1_1</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 2 -gpu </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>2_1_1</x>
    <replace_lines>
      <patches>  [2,1,1]   </patches>
    </replace_lines>
</Test>
<Test>
    <Title>3_1_1</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 3 -gpu </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>3_1_1</x>
    <replace_lines>
      <patches>  [3,1,1]   </patches>
    </replace_lines>
</Test>
<!--______________________________________________________________________-->
<Test>
    <Title>1_2_1</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 2 -gpu </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>1_2_1</x>
    <replace_lines>
      <patches>  [1,2,1]   </patches>
    </replace_lines>
</Test>

<Test>
    <Title>1_3_1</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 3 -gpu </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>1_3_1</x>
    <replace_lines>
      <patches>  [1,3,1]   </patches>
    </replace_lines>
</Test>

<!--______________________________________________________________________-->

<Test>
    <Title>1_1_2</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 2 -gpu </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>1_1_2</x>
    <replace_lines>
      <patches>  [1,1,2]   </patches>
    </replace_lines>
</Test>

<Test>
    <Title>1_1_3</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 3 -gpu </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>1_1_3</x>
    <replace_lines>
      <patches>  [1,1,3]   </patches>
    </replace_lines>
</Test>


<!--______________________________________________________________________-->

<Test>
    <Title>2_2_1</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 4 -gpu </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>2_2_1</x>
    <replace_lines>
      <patches>  [2,2,1]   </patches>
    </replace_lines>
</Test>

<Test>
    <Title>2_1_2</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 4 -gpu </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>2_1_2</x>
    <replace_lines>
      <patches>  [2,1,2]   </patches>
    </replace_lines>
</Test>

<Test>
    <Title>2_2_2</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 4 -gpu </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>2_2_2</x>
    <replace_lines>
      <patches>  [2,2,2]   </patches>
    </replace_lines>
</Test>

<!--______________________________________________________________________-->

<Test>
    <Title>3_3_1</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 9 -gpu </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>3_3_1</x>
    <replace_lines>
      <patches>  [3,3,1]   </patches>
    </replace_lines>
</Test>

<Test>
    <Title>3_1_3</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 9 -gpu </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>3_1_3</x>
    <replace_lines>
      <patches>  [3,1,3]   </patches>
    </replace_lines>
</Test>

<Test>
    <Title>3_3_3</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 9 -gpu </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>3_3_3</x>
    <replace_lines>
      <patches>  [3,3,3]   </patches>
    </replace_lines>
</Test>

</start>
