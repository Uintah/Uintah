<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>cavityFlow_dx.ups</upsFile>

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
    <postProcess_cmd>compare_cavityFlow.m -aDir 1 -mat 0 -plot true -Re 100</postProcess_cmd>
    <x>100</x>
    <replace_values>
      <entry path = "Uintah_specification/Grid/BoundaryConditions/Face[@side='y+']/BCType[@id = '0' and @label ='Velocity' and @var='Dirichlet']/value" value = '[0.02,0,0]' /> 
    </replace_values>
</Test>
<!--
<Test>
    <Title>400</Title>
    <sus_cmd>mpirun -np 6 sus </sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -aDir 1 -mat 0 -plot true -Re 400</postProcess_cmd>
    <x>400</x>
    <replace_values>
      <entry path = "Uintah_specification/Grid/BoundaryConditions/Face[@side='y+']/BCType[@id = '0' and @label ='Velocity' and @var='Dirichlet']/value" value = '[0.08,0,0]' /> 
    </replace_values>
</Test>
-->
<Test>
    <Title>1000</Title>
    <sus_cmd>mpirun -np 6 sus </sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -aDir 1 -mat 0 -plot true -Re 1000</postProcess_cmd>
    <x>1000</x>
    <replace_values>
      <entry path = "Uintah_specification/Grid/BoundaryConditions/Face[@side='y+']/BCType[@id = '0' and @label ='Velocity' and @var='Dirichlet']/value" value = '[.2,0,0]' /> 
    </replace_values>
</Test>
<!--
<Test>
    <Title>3200</Title>
    <sus_cmd>mpirun -np 6 sus </sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -aDir 1 -mat 0 -plot true -Re 3200</postProcess_cmd>
    <replace_values>
      <entry path = "Uintah_specification/Grid/BoundaryConditions/Face[@side='y+']/BCType[@id = '0' and @label ='Velocity' and @var='Dirichlet']/value" value = '[.64,0,0]' />
    </replace_values>
</Test>
<Test>
    <Title>5000</Title>
    <sus_cmd>mpirun -np 6 sus </sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -aDir 1 -mat 0 -plot true -Re 5000</postProcess_cmd>
    <replace_lines>
      <maxTime>   20    </maxTime>
    </replace_lines>
    <replace_values>
      <entry path = "Uintah_specification/Grid/BoundaryConditions/Face[@side='y+']/BCType[@id = '0' and @label ='Velocity' and @var='Dirichlet']/value" value = '[1,0,0]' />
    </replace_values>
</Test>
<Test>
    <Title>7500</Title>
    <sus_cmd>mpirun -np 6 sus </sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -aDir 1 -mat 0 -plot true -Re 7500</postProcess_cmd>
    <replace_lines>
      <maxTime>   15    </maxTime>
    </replace_lines>
    <replace_values>
      <entry path = "Uintah_specification/Grid/BoundaryConditions/Face[@side='y+']/BCType[@id = '0' and @label ='Velocity' and @var='Dirichlet']/value" value = '[1.5,0,0]' />
    </replace_values>
</Test>
<Test>
    <Title>10000</Title>
    <sus_cmd>mpirun -np 6 sus </sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -aDir 1 -mat 0 -plot true -Re 10000</postProcess_cmd>
    <replace_lines>
      <maxTime>   10    </maxTime>
    </replace_lines>
    <replace_values>
      <entry path = "Uintah_specification/Grid/BoundaryConditions/Face[@side='y+']/BCType[@id='0' and @label='Velocity' and @var='Dirichlet']/value" value = '[2,0,0]' /> 
    </replace_values>
</Test>
-->
</start>
