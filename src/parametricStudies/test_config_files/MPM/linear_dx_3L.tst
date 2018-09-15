<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>AA_MMS_3L.ups</upsFile>
<gnuplot>
  <script>plotScript.gp</script>
  <title>MPM:Axis Aligned MMS, 2Levels, 10 timeteps, Linear</title>
  <ylabel>L2 Error</ylabel>
  <xlabel>Resolution</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
    <interpolator>linear</interpolator>
    <extraCells>[0,0,0]' /></extraCells>
    <res>      [2,2,2]' />      </res>
  </replace_lines>
  <replace_values>
     <entry path="/Uintah_specification/Grid/Level/Box[@label=0]' />/lower" value = '[0.0, 0.0,  0.0]' />
     <entry path="/Uintah_specification/Grid/Level/Box[@label=0]' />/upper" value = '[1.0, 1.0,  1.0]' />

     <entry path="/Uintah_specification/Grid/Level/Box[@label=1]' />/lower" value = '[0.25, 0.25, 0.25]' />
     <entry path="/Uintah_specification/Grid/Level/Box[@label=1]' />/upper" value = '[0.75, 0.75, 0.75]' />
  </replace_values>
</AllTests>
<!--
<Test>
    <Title>8</Title>
    <sus_cmd> nice mpirun -np 1  sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm Linf -MMS 2 </postProcess_cmd>
    <x>8</x>
    <replace_values>
         <entry path="/Uintah_specification/Grid/Level/Box[@label=0]' />/resolution" value = '[8,8,8]' />
         <entry path="/Uintah_specification/Grid/Level/Box[@label=1]' />/resolution" value = '[4,4,4]' />
    </replace_values>
</Test>
-->
<Test>
    <Title>16</Title>
    <sus_cmd> nice mpirun -np 1  sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm Linf -MMS 2 </postProcess_cmd>
    <x>16</x>
    <replace_values>
         <entry path="/Uintah_specification/Grid/Level/Box[@label=0]' />/resolution" value = '[16,16,16]' />
         <entry path="/Uintah_specification/Grid/Level/Box[@label=1]' />/resolution" value = '[8,8,8]' />
    </replace_values>
</Test>

<Test>
    <Title>32</Title>
    <sus_cmd> nice mpirun -np 1  sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm Linf -MMS 2 </postProcess_cmd>
    <x>32</x>
    <replace_values>
         <entry path="/Uintah_specification/Grid/Level/Box[@label=0]' />/resolution" value = '[32,32,32]' />
         <entry path="/Uintah_specification/Grid/Level/Box[@label=1]' />/resolution" value = '[16,16,16]' />
    </replace_values>
</Test>

<Test>
    <Title>64</Title>
    <sus_cmd> nice mpirun -np 1  sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm Linf -MMS 2 </postProcess_cmd>
    <x>64</x>
    <replace_values>
         <entry path="/Uintah_specification/Grid/Level/Box[@label=0]' />/resolution" value = '[64,64,64]' />
         <entry path="/Uintah_specification/Grid/Level/Box[@label=1]' />/resolution" value = '[32,32,32]' />
    </replace_values>
</Test>
<!--
<Test>
    <Title>128</Title>
    <sus_cmd>nice mpirun -np 1  sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm Linf -MMS 2 </postProcess_cmd>
    <x>128</x>
    <replace_values>
         <entry path="/Uintah_specification/Grid/Level/Box[@label=0]' />/resolution" value = '[128,128,128]' />
         <entry path="/Uintah_specification/Grid/Level/Box[@label=1]' />/resolution" value = '[64,64,64]' />
    </replace_values>
</Test>
-->
</start>
