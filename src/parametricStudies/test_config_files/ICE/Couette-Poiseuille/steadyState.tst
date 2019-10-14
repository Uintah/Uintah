<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>CouettePoiseuille/XY.ups</upsFile>

<AllTests>
  <replace_lines>
    <spacing>         [0.02,0.004,0.01]  </spacing>
    <outputInterval>0.25</outputInterval>
  </replace_lines>
</AllTests>

<Test>
    <Title>0.5</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 0 -sliceDir 1 -mat 0 -P -0.25 -plot true</postProcess_cmd>
    <x>0.5</x>
    <replace_lines>
      <maxTime>  0.5   </maxTime>
    </replace_lines>
</Test>

<Test>
    <Title>1.0</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 0 -sliceDir 1 -mat 0 -P -0.25 -plot true</postProcess_cmd>
    <x>1.0</x>
    <replace_lines>
      <maxTime>  1.0   </maxTime>
    </replace_lines>
</Test>

<Test>
    <Title>2.0</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 0 -sliceDir 1 -mat 0 -P -0.25 -plot true</postProcess_cmd>
    <x>2.0</x>
    <replace_lines>
      <maxTime>  2.0   </maxTime>
    </replace_lines>
</Test>

<Test>
    <Title>4.0</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 0 -sliceDir 1 -mat 0 -P -0.25 -plot true</postProcess_cmd>
    <x>4.0</x>
    <replace_lines>
      <maxTime>  4.0   </maxTime>
    </replace_lines>
</Test>

<Test>
    <Title>6.0</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 0 -sliceDir 1 -mat 0 -P -0.25 -plot true</postProcess_cmd>
    <x>6.0</x>
    <replace_lines>
      <maxTime>  6.0   </maxTime>
    </replace_lines>
</Test>

<Test>
    <Title>8.0</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 0 -sliceDir 1 -mat 0 -P -0.25 -plot true</postProcess_cmd>
    <x>8.0</x>
    <replace_lines>
      <maxTime>  8.0   </maxTime>
    </replace_lines>
</Test>

</start>
