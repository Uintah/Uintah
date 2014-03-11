<start>
<upsFile>AA.ups</upsFile>
<gnuplot>
  <script>plotScript.gp</script>s
  <title>MPM: 1D Periodic Bar (z-dir), 1 Level, 10 timeteps, gimp</title>
  <ylabel>L2 Error</ylabel>
  <xlabel>Resolution</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
    <interpolator>gimp</interpolator>
    <extraCells>[1,1,0]</extraCells>
    <lower>[0.5,  0.5,  0.0]</lower>
    <upper>[0.501,0.501, 1.0]</upper>
    <periodic> [0,0,1] </periodic>
    <res>[2,2,2]</res>
  </replace_lines>
</AllTests>
<!--
<Test>
    <Title>8</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 1 </postProcess_cmd>
    <x>8</x>
    <replace_lines>
      <resolution>   [1, 1, 8]         </resolution>
      <upper>       [0.5125, 0.5125, 1.0]</upper>
    </replace_lines>
</Test>
-->
<Test>
    <Title>16</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 1 </postProcess_cmd>
    <x>16</x>
    <replace_lines>
      <resolution>   [1, 1, 16]         </resolution>
      <upper>       [0.50625, 0.50625, 1.0]</upper>
    </replace_lines>
</Test>

<Test>
    <Title>32</Title>
    <sus_cmd>nice mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 1 </postProcess_cmd>
    <x>32</x>
    <replace_lines>
      <resolution>   [1, 1, 32]         </resolution>
      <upper>       [0.503125, 0.503125, 1.0]</upper>
    </replace_lines>
</Test>

<Test>
    <Title>64</Title>
    <sus_cmd>nice mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 1 </postProcess_cmd>
    <x>64</x>
    <replace_lines>
      <resolution>   [1, 1, 64]         </resolution>
      <upper>       [0.501563, 0.501563, 1.0]</upper>
    </replace_lines>
</Test>

<Test>
    <Title>128</Title>
    <sus_cmd>nice mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 1 </postProcess_cmd>
    <x>128</x>
    <replace_lines>
      <resolution>   [1, 1,128]      </resolution>
      <upper>       [0.5007813, 0.5007813, 1.0]</upper>
    </replace_lines>
</Test>

<Test>
    <Title>256</Title>
    <sus_cmd>nice mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 1 </postProcess_cmd>
    <x>256</x>
    <replace_lines>
      <resolution>   [1, 1, 256]      </resolution>
      <upper>       [0.5003906, 0.5003906, 1.0]</upper>
    </replace_lines>
</Test>

<Test>
    <Title>512</Title>
    <sus_cmd>nice mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 1 </postProcess_cmd>
    <x>512</x>
    <replace_lines>
      <resolution>   [1, 1, 512]      </resolution>
      <upper>       [0.50019531, 0.50019531, 1.0]</upper>
    </replace_lines>
</Test>

</start>
