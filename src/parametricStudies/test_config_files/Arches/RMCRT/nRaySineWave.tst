<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>RMCRT_sineWave.ups</upsFile>
<gnuplot>
  <script>plotScript_noFit.gp</script>
  <title>Double:CPU:RMCRT\\n Benchmark #5 \\n 41^3, Random Seed</title>
  <ylabel>L2 norm Error</ylabel>
  <xlabel># of Rays</xlabel>
</gnuplot>

<AllTests>
    <max_Timesteps>1</max_Timesteps>
    <randomSeed> true </randomSeed>
</AllTests>


<Test>
    <Title>8</Title>
    <sus_cmd> mpirun -np 4 sus -mpi </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 5 -L 0 -plot false</postProcess_cmd>
    <x>8</x>
    <replace_lines>
      <nDivQRays>          8        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>16</Title>
    <sus_cmd> mpirun -np 4 sus -mpi </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 5 -L 0 -plot false</postProcess_cmd>
    <x>16</x>
    <replace_lines>
      <nDivQRays>          16        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>32</Title>
    <sus_cmd> mpirun -np 4 sus -mpi </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 5 -L 0 -plot false</postProcess_cmd>
    <x>32</x>
    <replace_lines>
      <nDivQRays>          32        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>64</Title>
    <sus_cmd> mpirun -np 4 sus -mpi </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 5 -L 0 -plot false</postProcess_cmd>
    <x>64</x>
    <replace_lines>
      <nDivQRays>          64        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>128</Title>
    <sus_cmd> mpirun -np 4 sus -mpi </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 5 -L 0 -plot false</postProcess_cmd>
    <x>128</x>
    <replace_lines>
      <nDivQRays>          128        </nDivQRays>
    </replace_lines>
</Test>

</start>
