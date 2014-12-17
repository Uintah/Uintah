<start>
<upsFile>advectPS.ups</upsFile>
<gnuplot>
  <script>plotScript.gp</script>s
  <title>ICE:Advection Test X dir</title>
  <ylabel>Error</ylabel>
  <xlabel>Resolution</xlabel>
</gnuplot>


<Test>
    <Title>100</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_scalar -v</postProcess_cmd>
    <x>100</x>
    <replace_lines>
       <delt_init>   2.0e-5             </delt_init>
      <resolution>   [100,1,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>200</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_scalar -v</postProcess_cmd>
    <x>200</x>
    <replace_lines>
      <delt_init>    1.0e-5             </delt_init>
      <resolution>   [200,1,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>400</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_scalar -v</postProcess_cmd>
    <x>400</x>
    <replace_lines>
      <delt_init>    5.0e-6             </delt_init>
      <resolution>   [400,1,1]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>800</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_scalar -v</postProcess_cmd>
    <x>800</x>
    <replace_lines>
      <delt_init>    2.5e-6             </delt_init>
      <resolution>   [800,1,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>1600</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_scalar -v</postProcess_cmd>
    <x>1600</x>
    <replace_lines>
      <delt_init>    1.25e-6             </delt_init>
      <resolution>   [1600,1,1]          </resolution>
    </replace_lines>
</Test>

</start>
