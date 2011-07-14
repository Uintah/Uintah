<start>
<upsFile>rayleigh.ups</upsFile>

<gnuplot>
  <script>plotScript.gp</script>s
  <title>ICE:Rayleigh Problem Y dir</title>
  <ylabel>Error</ylabel>
  <xlabel>Resolution</xlabel>
</gnuplot>

<AllTests>
</AllTests>
<Test>
    <Title>25</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -pDir 1 -mat 0 -plot false</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <resolution>   [10,25,1]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>50</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -pDir 1 -mat 0 -plot false</postProcess_cmd>
    <x>50</x>
    <replace_lines>
      <resolution>   [10,50,1]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>100</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -pDir 1 -mat 0 -plot false</postProcess_cmd>
    <x>100</x>
    <replace_lines>
      <resolution>   [10,100,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>200</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -pDir 1 -mat 0 -plot false</postProcess_cmd>
    <x>200</x>
    <replace_lines>
      <resolution>   [10,200,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>400</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -pDir 1 -mat 0 -plot false</postProcess_cmd>
    <x>400</x>
    <replace_lines>
      <resolution>   [10,400,1]          </resolution>
    </replace_lines>
</Test>

</start>
