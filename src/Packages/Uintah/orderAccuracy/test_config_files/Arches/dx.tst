<start>
<upsFile>almgrenMMS.ups</upsFile>
<Study>Res.Study</Study>
<gnuplot>
  <script>plotScript.gp</script>s
  <title>Arches:MMS:almgren X dir</title>
  <ylabel>Error</ylabel>
  <xlabel>Resolution</xlabel>
</gnuplot>
<Test>
    <Title>8</Title>
    <sus_cmd>mpirun -np 1 sus -arches </sus_cmd>
    <Study>Res.Study</Study>
    <compare_cmd>arches_mms.m -pDir 1  </compare_cmd>
    <x>8</x>
    <replace_lines>
      <resolution>   [8,8,8]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>16</Title>
    <sus_cmd>mpirun -np 1 sus -arches </sus_cmd>
    <Study>Res.Study</Study>
    <compare_cmd>arches_mms.m -pDir 1  </compare_cmd>
    <x>16</x>
    <replace_lines>
      <resolution>   [16,16,16]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>32</Title>
    <sus_cmd>mpirun -np 1 sus -arches </sus_cmd>
    <Study>Res.Study</Study>
    <compare_cmd>arches_mms.m -pDir 1 </compare_cmd>
    <x>32</x>
    <replace_lines>
      <resolution>   [32,32,32]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>64</Title>
    <sus_cmd>mpirun -np 1 sus -arches </sus_cmd>
    <Study>Res.Study</Study>
    <compare_cmd>arches_mms.m -pDir 1 </compare_cmd>
    <x>64</x>
    <replace_lines>
      <resolution>   [64,64,64]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>128</Title>
    <sus_cmd>mpirun -np 1 sus -arches </sus_cmd>
    <Study>Res.Study</Study>
    <compare_cmd>arches_mms.m -pDir 1 </compare_cmd>
    <x>128</x>
    <replace_lines>
      <resolution>   [128,128,128]          </resolution>
    </replace_lines>
</Test>

</start>
