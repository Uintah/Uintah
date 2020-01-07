<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>CouettePoiseuille/XY.ups</upsFile>

<gnuplot>
  <script>plotScript.gp</script>
  <title>ICE: Couette Poiseuille Problem (P = -0.25, U = 13.781 dpdx = -100 )</title>
  <ylabel>Error</ylabel>
  <xlabel>DelT knob (diffusion)</xlabel>
  <label> "Time: 1.0s, CFL: 0.4\\nAdvection: 2nd order scheme\\nSemi-implicit pressure\\n20 cells in axial direction, 1 cell 'deep'" at graph 0.01,0.1 font "Times,10" </label>
</gnuplot>

<AllTests>
  <replace_lines>
    <maxTime>  1.0   </maxTime>
    <outputInterval>0.25 </outputInterval>
    <knob_for_speedSound>  0.0  </knob_for_speedSound>
    <spacing>         [0.02,0.0025,0.01]  </spacing>
    <patches>      [1,1,1]                </patches>   
  </replace_lines>
</AllTests>

<Test>
    <Title>0.0</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 0 -sliceDir 1 -mat 0 -P -0.25 -plot true</postProcess_cmd>
    <x>0.0</x>
    <replace_lines>
      <knob_for_diffusion>   0.0  </knob_for_diffusion>
    </replace_lines>
</Test>

<Test>
    <Title>0.2</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 0 -sliceDir 1 -mat 0 -P -0.25 -plot true</postProcess_cmd>
    <x>0.2</x>
    <replace_lines>
      <knob_for_diffusion>   0.2  </knob_for_diffusion>
    </replace_lines>
</Test>

<Test>
    <Title>0.4</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 0 -sliceDir 1 -mat 0 -P -0.25 -plot true</postProcess_cmd>
    <x>0.4</x>
    <replace_lines>
      <knob_for_diffusion>   0.4  </knob_for_diffusion>
    </replace_lines>
</Test>

<Test>
    <Title>0.5</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 0 -sliceDir 1 -mat 0 -P -0.25 -plot true</postProcess_cmd>
    <x>0.5</x>
    <replace_lines>
      <knob_for_diffusion>   0.5  </knob_for_diffusion>
    </replace_lines>
</Test>

<Test>
    <Title>0.6</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 0 -sliceDir 1 -mat 0 -P -0.25 -plot true</postProcess_cmd>
    <x>0.6</x>
    <replace_lines>
      <knob_for_diffusion>   0.6  </knob_for_diffusion>
    </replace_lines>
</Test>

<Test>
    <Title>0.7</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 0 -sliceDir 1 -mat 0 -P -0.25 -plot true</postProcess_cmd>
    <x>0.7</x>
    <replace_lines>
      <knob_for_diffusion>   0.7  </knob_for_diffusion>
    </replace_lines>
</Test>

<Test>
    <Title>0.8</Title>
    <sus_cmd> mpirun sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 0 -sliceDir 1 -mat 0 -P -0.25 -plot true</postProcess_cmd>
    <x>0.8</x>
    <replace_lines>
     <knob_for_diffusion>   0.8  </knob_for_diffusion>
    </replace_lines>
</Test>

<Test>
    <Title>1.0</Title>
    <sus_cmd> mpirun sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 0 -sliceDir 1 -mat 0 -P -0.25 -plot true</postProcess_cmd>
    <x>1.0</x>
    <replace_lines>
     <knob_for_diffusion>   1.0  </knob_for_diffusion>
    </replace_lines>
</Test>


</start>
