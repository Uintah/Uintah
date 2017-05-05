<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>AA.ups</upsFile>
<gnuplot>
  <script>plotScript.gp</script>
  <title>MPM:Axis Aligned MMS, 10 timeteps, 3rdorderBS</title>
  <ylabel>L2 Error</ylabel>
  <xlabel>Resolution</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
    <interpolator>3rdorderBS</interpolator>
    <extraCells>[1,1,1]</extraCells>
  </replace_lines>
</AllTests>

<Test>
    <Title>8</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 2 </postProcess_cmd>
    <x>8</x>
    <replace_lines>
      <resolution>   [8,8,8]         </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>16</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 2 </postProcess_cmd>
    <x>16</x>
    <replace_lines>
      <resolution>   [16,16,16]         </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>32</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>compare_MPM_AA_MMS.m -norm L2 -MMS 2 </postProcess_cmd>
    <x>32</x>
    <replace_lines>
      <resolution>   [32,32,32]         </resolution>
    </replace_lines>
</Test>

</start>
