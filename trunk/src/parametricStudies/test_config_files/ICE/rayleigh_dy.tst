<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>rayleigh_dy.ups</upsFile>

<gnuplot>
  <script>plotScript.gp</script>
  <title>ICE:Rayleigh Problem Y direction</title>
  <ylabel>Error</ylabel>
  <xlabel>Resolution</xlabel>
  <label> "Time: 0.2s, CFL: 0.4\\nAdvection: 2nd order scheme\\nSemi-implicit pressure\\n20 cells in axial direction, 1 cell 'deep'" at graph 0.01,0.1 font "Times,10" </label>
</gnuplot>

<AllTests>
  <replace_lines>
    <maxTime>  0.2   </maxTime>
    <outputInterval>0.1</outputInterval>
    <upper>        [0.01,0.5,0.025]    </upper>
  </replace_lines>
</AllTests>

<Test>
    <Title>10</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 2 -mat 0 -plot true</postProcess_cmd>
    <x>10</x>
    <replace_lines>
      <resolution>   [1,20,10]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>25</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 2 -mat 0 -plot true</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <resolution>   [1,20,25]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>50</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 2 -mat 0 -plot true</postProcess_cmd>
    <x>50</x>
    <replace_lines>
      <resolution>   [1,20,50]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>75</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 2 -mat 0 -plot true</postProcess_cmd>
    <x>75</x>
    <replace_lines>
      <resolution>   [1,20,75]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>100</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 2 -mat 0 -plot true</postProcess_cmd>
    <x>100</x>
    <replace_lines>
      <resolution>   [1,20,100]          </resolution>
    </replace_lines>
</Test>

<!--
<Test>
    <Title>200</Title>
    <sus_cmd>mpirun -np 2 sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 2 -mat 0 -plot true</postProcess_cmd>
    <x>200</x>
    <replace_lines>
      <resolution>   [1,20,200]          </resolution>
      <patches>      [1,1,2]            </patches>
    </replace_lines>
</Test>

<Test>
    <Title>400</Title>
    <sus_cmd> mpirun -np 4 sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 2 -mat 0 -plot true</postProcess_cmd>
    <x>400</x>
    <replace_lines>
      <resolution>   [1,20,400]          </resolution>
      <patches>      [1,1,4]            </patches>
    </replace_lines>
</Test>
-->
</start>
