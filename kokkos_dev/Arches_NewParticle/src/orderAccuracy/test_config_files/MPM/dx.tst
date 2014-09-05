<start>
<upsFile>AA.ups</upsFile>
<Study>Res.Study</Study>
<gnuplot>
  <script>plotScript.gp</script>s
  <title>MPM:Axis Alignmed MMS, 10 timeteps</title>
  <ylabel>Error</ylabel>
  <xlabel>Resolution</xlabel>
</gnuplot>


<Test>
    <Title>8</Title>
    <sus_cmd> sus </sus_cmd>
    <Study>Res.Study</Study>
    <postProcess_cmd>compare_MPM_AA_MMS.m</postProcess_cmd>
    <x>8</x>
    <replace_lines>
      <resolution>   [8,8,8]         </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>16</Title>
    <sus_cmd> sus </sus_cmd>
    <Study>Res.Study</Study>
    <postProcess_cmd>compare_MPM_AA_MMS.m</postProcess_cmd>
    <x>16</x>
    <replace_lines>
      <resolution>   [16,16,16]         </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>32</Title>
    <sus_cmd>mpirun -np 8 sus </sus_cmd>
    <Study>Res.Study</Study>
    <postProcess_cmd>compare_MPM_AA_MMS.m</postProcess_cmd>
    <x>32</x>
    <replace_lines>
      <patches>      [2,2,2]            </patches>
      <resolution>   [32,32,32]         </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>64</Title>
    <sus_cmd>mpirun -np 8 sus </sus_cmd>
    <Study>Res.Study</Study>
    <postProcess_cmd>compare_MPM_AA_MMS.m</postProcess_cmd>
    <x>64</x>
    <replace_lines>
      <patches>      [2,2,2]            </patches>
      <resolution>   [64,64,64]         </resolution>
    </replace_lines>
</Test>
</start>
