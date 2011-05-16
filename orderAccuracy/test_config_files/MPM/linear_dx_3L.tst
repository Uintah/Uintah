<start>
<upsFile>AA_MMS_3L.ups</upsFile>
<Study>Res.Study</Study>
<gnuplot>
  <script>plotScript.gp</script>
  <title>MPM:Axis Aligned MMS, 2Levels, 10 timeteps, Linear</title>
  <ylabel>L2 Error</ylabel>
  <xlabel>Resolution</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
    <interpolator>linear</interpolator>
    <extraCells>[0,0,0]</extraCells>
  </replace_lines>
</AllTests>

<Test>
    <Title>8</Title>
    <sus_cmd> nice mpirun -np 1  sus </sus_cmd>
    <Study>Res.Study</Study>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 </postProcess_cmd>
    <x>8</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/resolution :[16,16,1]
         /Uintah_specification/Grid/Level/Box[@label=1]/resolution :[8,8,1]
    </replace_values>
</Test>

<Test>
    <Title>16</Title>
    <sus_cmd> nice mpirun -np 1  sus </sus_cmd>
    <Study>Res.Study</Study>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 </postProcess_cmd>
    <x>16</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/resolution :[32,32,1]
         /Uintah_specification/Grid/Level/Box[@label=1]/resolution :[16,16,1]
    </replace_values>
</Test>

<Test>
    <Title>32</Title>
    <sus_cmd> nice mpirun -np 1  sus </sus_cmd>
    <Study>Res.Study</Study>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 </postProcess_cmd>
    <x>32</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/resolution :[64,64,1]
         /Uintah_specification/Grid/Level/Box[@label=1]/resolution :[32,32,1]
    </replace_values>
</Test>

<Test>
    <Title>64</Title>
    <sus_cmd>nice mpirun -np 1  sus </sus_cmd>
    <Study>Res.Study</Study>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 </postProcess_cmd>
    <x>64</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/resolution :[128,128,1]
         /Uintah_specification/Grid/Level/Box[@label=1]/resolution :[64,64,1]
    </replace_values>
</Test>
<!--
<Test>
    <Title>96</Title>
    <sus_cmd>nice mpirun -np 1 sus </sus_cmd>
    <Study>Res.Study</Study>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 </postProcess_cmd>
    <x>96</x>
    <replace_lines>
      <patches>      [2,2,2]            </patches>
      <resolution>   [96,96,96]      </resolution>
    </replace_lines>
</Test>
-->
</start>
