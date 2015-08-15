<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>rayleigh_dy.ups</upsFile>

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
    <postProcess_cmd>compare_Rayleigh.m -aDir 2 -mat 0 -plot false</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <resolution>   [1,10,25]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>50</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 2 -mat 0 -plot false</postProcess_cmd>
    <x>50</x>
    <replace_lines>
      <resolution>   [1,10,50]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>100</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 2 -mat 0 -plot false</postProcess_cmd>
    <x>100</x>
    <replace_lines>
      <resolution>   [1,10,100]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>200</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 2 -mat 0 -plot false</postProcess_cmd>
    <x>200</x>
    <replace_lines>
      <resolution>   [1,10,200]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>400</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 2 -mat 0 -plot false</postProcess_cmd>
    <x>400</x>
    <replace_lines>
      <resolution>   [1,10,400]          </resolution>
    </replace_lines>
</Test>

</start>
