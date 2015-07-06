<start>
<upsFile>RMCRT_xkexp.ups</upsFile>
<gnuplot>
  <script>plotScript.gp</script>s
  <title>Double:CPU:RMCRT Order-Of-Accuracy \\n Burns and Christon Benchmark \\n 1  41^3, Random Seed</title>
  <ylabel>Error</ylabel>
  <xlabel># of Rays</xlabel>
</gnuplot>

<AllTests>
    <max_Timesteps>40</max_Timesteps>
    <randomSeed> true </randomSeed>
</AllTests>


<Test>
    <Title>0400</Title>
    <sus_cmd> mpirun -np 4 sus -mpi </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 6 -L 0 -plot true</postProcess_cmd>
    <x>24</x>
    <replace_lines>
      <nDivQRays>          10        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>010</Title>
    <sus_cmd> mpirun -np 4 sus -mpi </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 6 -L 0 -plot true</postProcess_cmd>
    <x>24</x>
    <replace_lines>
      <nDivQRays>          20        </nDivQRays>
    </replace_lines>
</Test>

</start>
