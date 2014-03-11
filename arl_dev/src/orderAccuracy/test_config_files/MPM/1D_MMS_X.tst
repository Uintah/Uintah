<start>
<upsFile>AA.ups</upsFile>
<gnuplot>
  <script>plotScript.gp</script>s
  <title>MPM: 1D Periodic Bar (x-dir), 1 Level, 10 timeteps, gimp</title>
  <ylabel>L2 Error</ylabel>
  <xlabel>Resolution</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
    <interpolator>gimp</interpolator>
    <extraCells>[0,1,1]</extraCells>
    <lower>[0.0,0.5,  0.5]</lower>
    <upper>[1.0,0.501,0.501]</upper>
    <res>[2,2,2]</res>
    <periodic> [1,0,0] </periodic>
  </replace_lines>
</AllTests>
<!--
<Test>
    <Title>8</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 1 </postProcess_cmd>
    <x>8</x>
    <replace_lines>
      <resolution>   [8,1,1]         </resolution>
      <upper>       [1.0,0.5125,0.5125]</upper>
    </replace_lines>
</Test>
-->
<Test>
    <Title>16</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 1 </postProcess_cmd>
    <x>16</x>
    <replace_lines>
      <resolution>   [16,1,1]         </resolution>
      <upper>       [1.0,0.50625,0.50625]</upper>
    </replace_lines>
</Test>

<Test>
    <Title>32</Title>
    <sus_cmd>nice mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 1 </postProcess_cmd>
    <x>32</x>
    <replace_lines>
      <resolution>   [32,1,1]         </resolution>
      <upper>       [1.0,0.503125,0.503125]</upper>
    </replace_lines>
</Test>

<Test>
    <Title>64</Title>
    <sus_cmd>nice mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 1 </postProcess_cmd>
    <x>64</x>
    <replace_lines>
      <resolution>   [64,1,1]         </resolution>
      <upper>       [1.0,0.501563,0.501563]</upper>
    </replace_lines>
</Test>

<Test>
    <Title>128</Title>
    <sus_cmd>nice mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 1 </postProcess_cmd>
    <x>128</x>
    <replace_lines>
      <resolution>   [128,1,1]      </resolution>
      <upper>       [1.0,0.5007813,0.5007813]</upper>
    </replace_lines>
</Test>

<Test>
    <Title>256</Title>
    <sus_cmd>nice mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 1 </postProcess_cmd>
    <x>256</x>
    <replace_lines>
      <resolution>   [256,1,1]      </resolution>
      <upper>       [1.0,0.5003906,0.5003906]</upper>
    </replace_lines>
</Test>

<Test>
    <Title>512</Title>
    <sus_cmd>nice mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 1 </postProcess_cmd>
    <x>512</x>
    <replace_lines>
      <resolution>   [512,1,1]      </resolution>
      <upper>       [1.0,0.50019531,0.50019531]</upper>
    </replace_lines>
</Test>

</start>
