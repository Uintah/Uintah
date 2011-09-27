<start>
<upsFile>AA.ups</upsFile>
<gnuplot>
  <script>plotScript.gp</script>s
  <title>MPM:Axis Aligned MMS, 1 Level, 10 timeteps, gimp</title>
  <ylabel>L2 Error</ylabel>
  <xlabel>Resolution</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
    <interpolator>gimp</interpolator>
    <extraCells>[1,1,1]</extraCells>
    <lower>[0.0, 0.0, 0.0]</lower>
    <upper>[1.0, 1.0, 1.0]</upper>
    <res>      [2,2,2]      </res>
  </replace_lines>
</AllTests>

<Test>
    <Title>8</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 2</postProcess_cmd>
    <x>8</x>
    <replace_lines>
      <resolution>   [8,8,8]         </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>16</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 2</postProcess_cmd>
    <x>16</x>
    <replace_lines>
      <resolution>   [16,16,16]         </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>32</Title>
    <sus_cmd>nice mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 2</postProcess_cmd>
    <x>32</x>
    <replace_lines>
      <patches>      [1,1,1]            </patches>
      <resolution>   [32,32,32]         </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>64</Title>
    <sus_cmd>nice mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 2</postProcess_cmd>
    <x>64</x>
    <replace_lines>
      <patches>      [2,2,2]            </patches>
      <resolution>   [64,64,64]         </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>96</Title>
    <sus_cmd>nice mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 2</postProcess_cmd>
    <x>96</x>
    <replace_lines>
      <patches>      [2,2,2]            </patches>
      <resolution>   [96,96,96]         </resolution>
    </replace_lines>
</Test>

<!--
<Test>
    <Title>128</Title>
    <sus_cmd>nice mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 2</postProcess_cmd>
    <x>128</x>
    <replace_lines>
      <patches>      [1,1,1]            </patches>
      <resolution>   [128,128,128]      </resolution>
    </replace_lines>
</Test>
-->
</start>
