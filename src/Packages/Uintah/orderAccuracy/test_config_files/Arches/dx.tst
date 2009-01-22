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
    <Title>16</Title>
    <sus_cmd>mpirun -np 1 sus -arches </sus_cmd>
    <Study>Res.Study</Study>
    <x>16</x>
    <replace_lines>
       <delt_init>   2.0e-5             </delt_init>
      <resolution>   [16,16,16]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>32</Title>
    <sus_cmd>mpirun -np 1 sus -arches </sus_cmd>
    <Study>Res.Study</Study>
    <x>32</x>
    <replace_lines>
      <delt_init>    1.0e-5             </delt_init>
      <resolution>   [32,16,16]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>64</Title>
    <sus_cmd>mpirun -np 1 sus -arches </sus_cmd>
    <Study>Res.Study</Study>
    <x>64</x>
    <replace_lines>
      <delt_init>    5.0e-6             </delt_init>
      <resolution>   [64,16,16]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>128</Title>
    <sus_cmd>mpirun -np 1 sus -arches </sus_cmd>
    <Study>Res.Study</Study>
    <x>128</x>
    <replace_lines>
      <delt_init>    2.5e-6             </delt_init>
      <resolution>   [128,16,16]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>256</Title>
    <sus_cmd>mpirun -np 1 sus -arches </sus_cmd>
    <Study>Res.Study</Study>
    <x>256</x>
    <replace_lines>
      <delt_init>    1.25e-6             </delt_init>
      <resolution>   [256,16,16]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>512</Title>
    <sus_cmd>mpirun -np 1 sus -arches </sus_cmd>
    <Study>Res.Study</Study>
    <x>512</x>
    <replace_lines>
      <delt_init>    6.25e-7             </delt_init>
      <resolution>   [512,16,16]          </resolution>
    </replace_lines>
</Test>

</start>
