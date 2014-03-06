<start>
<upsFile>almgrenMMS.ups</upsFile>
<gnuplot>
  <script>plotScript.gp</script>s
  <title>Arches:MMS:almgren:Spatial order-of-accuracy \\n 1 timestep,dt = 1e-5</title>
  <ylabel>Error</ylabel>
  <xlabel>Resolution</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
   <timestep_multiplier>1</timestep_multiplier>
   <delt_init>1e-5</delt_init>
  </replace_lines>
</AllTests>

<Test>
    <Title>8</Title>
    <sus_cmd>mpirun -np 1 sus -mpi</sus_cmd>
    <postProcess_cmd>arches_mms.m </postProcess_cmd>
    <x>8</x>
    <replace_lines>
      <resolution>   [8,8,8]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>16</Title>
    <sus_cmd>mpirun -np 1 sus -mpi</sus_cmd>
    <postProcess_cmd>arches_mms.m  </postProcess_cmd>
    <x>16</x>
    <replace_lines>
      <resolution>   [16,16,16]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>32</Title>
    <sus_cmd>mpirun -np 1 sus -mpi</sus_cmd>
    <postProcess_cmd>arches_mms.m </postProcess_cmd>
    <x>32</x>
    <replace_lines>
      <resolution>   [32,32,32]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>64</Title>
    <sus_cmd>mpirun -np 1 sus -mpi</sus_cmd>
    <postProcess_cmd>arches_mms.m </postProcess_cmd>
    <x>64</x>
    <replace_lines>
      <resolution>   [64,64,64]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>128</Title>
    <sus_cmd>mpirun -np 1 sus -mpi</sus_cmd>
    <postProcess_cmd>arches_mms.m </postProcess_cmd>
    <x>128</x>
    <replace_lines>
      <resolution>   [128,128,128]          </resolution>
    </replace_lines>
</Test>

</start>
