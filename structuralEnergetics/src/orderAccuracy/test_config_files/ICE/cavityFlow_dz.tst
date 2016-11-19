<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>cavityFlow_dz.ups</upsFile>

<gnuplot>
  <script>plotScript.gp</script>s
  <title>ICE:Cavity Flow x Dir</title>
  <ylabel>Error</ylabel>
  <xlabel>Resolution</xlabel>
</gnuplot>

<AllTests>
</AllTests>
<Test>
    <Title>100</Title>
    <sus_cmd>mpirun -np 6 sus </sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -aDir 3 -mat 0 -plot true -Re 100</postProcess_cmd>
    <x>100</x>
    <replace_values>
      /Uintah_specification/Grid/BoundaryConditions/Face[@side='x+']/BCType[@id = '0' and @label ='Velocity' and @var='Dirichlet']/value : [0,0,0.02] 
    </replace_values>
</Test>
<!--
<Test>
    <Title>400</Title>
    <sus_cmd>mpirun -np 6 sus </sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -aDir 3 -mat 0 -plot true -Re 400</postProcess_cmd>
    <x>400</x>
    <replace_values>
      /Uintah_specification/Grid/BoundaryConditions/Face[@side='x+']/BCType[@id = '0' and @label ='Velocity' and @var='Dirichlet']/value : [0,0,0.08] 
    </replace_values>
</Test>
-->
<Test>
    <Title>1000</Title>
    <sus_cmd>mpirun -np 6 sus </sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -aDir 3 -mat 0 -plot true -Re 1000</postProcess_cmd>
    <x>1000</x>
    <replace_values>
      /Uintah_specification/Grid/BoundaryConditions/Face[@side='x+']/BCType[@id = '0' and @label ='Velocity' and @var='Dirichlet']/value : [0,0,.2] 
    </replace_values>
</Test>
<!--
<Test>
    <Title>3200</Title>
    <sus_cmd>mpirun -np 6 sus </sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -aDir 3 -mat 0 -plot true -Re 3200</postProcess_cmd>
    <replace_values>
      /Uintah_specification/Grid/BoundaryConditions/Face[@side='x+']/BCType[@id = '0' and @label ='Velocity' and @var='Dirichlet']/value : [0,0,.64]
    </replace_values>
</Test>
<Test>
    <Title>5000</Title>
    <sus_cmd>mpirun -np 6 sus </sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -aDir 3 -mat 0 -plot true -Re 5000</postProcess_cmd>
    <replace_lines>
      <maxTime>   20    </maxTime>
    </replace_lines>
    <replace_values>
      /Uintah_specification/Grid/BoundaryConditions/Face[@side='x+']/BCType[@id = '0' and @label ='Velocity' and @var='Dirichlet']/value : [0,0,1]
    </replace_values>
</Test>
<Test>
    <Title>7500</Title>
    <sus_cmd>mpirun -np 6 sus </sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -aDir 3 -mat 0 -plot true -Re 7500</postProcess_cmd>
    <replace_lines>
      <maxTime>   15    </maxTime>
    </replace_lines>
    <replace_values>
      /Uintah_specification/Grid/BoundaryConditions/Face[@side='x+']/BCType[@id = '0' and @label ='Velocity' and @var='Dirichlet']/value : [0,0,1.5]
    </replace_values>
</Test>
<Test>
    <Title>10000</Title>
    <sus_cmd>mpirun -np 6 sus </sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -aDir 3 -mat 0 -plot true -Re 10000</postProcess_cmd>
    <replace_lines>
      <maxTime>   10    </maxTime>
    </replace_lines>
    <replace_values>
      /Uintah_specification/Grid/BoundaryConditions/Face[@side='x+']/BCType[@id='0' and @label='Velocity' and @var='Dirichlet']/value : [0,0,2] 
    </replace_values>
</Test>
-->
</start>
