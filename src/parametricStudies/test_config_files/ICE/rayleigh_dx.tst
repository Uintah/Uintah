<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>rayleigh_dx.ups</upsFile>

<gnuplot>
  <script>plotScript.gp</script>s
  <title>ICE:Rayleigh Problem X direction</title>
  <ylabel>Error</ylabel>
  <xlabel>Resolution</xlabel>
  <label> "Time: 0.2s, CFL: 0.4\\nAdvection: 2nd order scheme\\nSemi-implicit pressure\\n20 cells in axial direction, 1 cell 'deep'" at graph 0.01,0.1 font "Times,10" </label>
</gnuplot>

<AllTests>
  <replace_lines>
    <maxTime>  0.2   </maxTime>
    <outputInterval>0.1</outputInterval>
    <upper>        [0.5,0.025,0.01]    </upper>
  </replace_lines>
</AllTests>

<Test>
    <Title>10</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 1 -mat 0 -plot true</postProcess_cmd>
    <x>10</x>
    <replace_lines>
      <resolution>   [20,10,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>25</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 1 -mat 0 -plot true</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <resolution>   [20,25,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>50</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 1 -mat 0 -plot true</postProcess_cmd>
    <x>50</x>
    <replace_lines>
      <resolution>   [20,50,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>75</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 1 -mat 0 -plot true</postProcess_cmd>
    <x>75</x>
    <replace_lines>
      <resolution>   [20,75,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>100</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 1 -mat 0 -plot true</postProcess_cmd>
    <x>100</x>
    <replace_lines>
      <resolution>   [20,100,1]          </resolution>
    </replace_lines>
</Test>

<!--
<Test>
    <Title>200</Title>
    <sus_cmd>mpirun -np 2 sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 1 -mat 0 -plot true</postProcess_cmd>
    <x>200</x>
    <replace_lines>
      <resolution>   [20,200,1]         </resolution>
      <patches>      [1,2,1]            </patches>
    </replace_lines>
</Test>

<Test>
    <Title>400</Title>
    <sus_cmd>mpirun -np 4 sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -aDir 1 -mat 0 -plot true</postProcess_cmd>
    <x>400</x>
    <replace_lines>
      <resolution>   [20,400,1]          </resolution>
      <patches>      [1,4,1]             </patches>
    </replace_lines>
</Test>
-->
</start>
