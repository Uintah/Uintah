<start>
<upsFile>test3.ups</upsFile>

<gnuplot>
  <script>plotRiemannTests.gp</script>
  <title>ICE: Toro's Test problem 3  X-direction</title>
  <ylabel>Error</ylabel>
  <xlabel>Resolution</xlabel>
</gnuplot>

<AllTests>
</AllTests>
<Test>
    <Title>100</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Riemann.m -test 3 -pDir 1 -var press_CC -mat 0</postProcess_cmd>
    <x>100</x>
    <replace_lines>
      <resolution>   [100,1,1]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>200</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Riemann.m -test 3 -pDir 1 -var press_CC -mat 0</postProcess_cmd>
    <x>200</x>
    <replace_lines>
      <resolution>   [200,1,1]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>400</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Riemann.m -test 3 -pDir 1 -var press_CC -mat 0</postProcess_cmd>
    <x>400</x>
    <replace_lines>
      <resolution>   [400,1,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>800</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Riemann.m -test 3 -pDir 1 -var press_CC -mat 0</postProcess_cmd>
    <x>800</x>
    <replace_lines>
      <resolution>   [800,1,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>1600</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Riemann.m -test 3 -pDir 1 -var press_CC -mat 0</postProcess_cmd>
    <x>1600</x>
    <replace_lines>
      <resolution>   [1600,1,1]          </resolution>
    </replace_lines>
</Test>

</start>
