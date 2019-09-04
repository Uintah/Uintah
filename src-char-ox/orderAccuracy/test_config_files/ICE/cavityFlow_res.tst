<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>cavityFlow_dx.ups</upsFile>

<gnuplot>
  <script>plotScript.gp</script>s
  <title>ICE:Cavity Flow Res</title>
  <ylabel>Error</ylabel>
  <xlabel>Resolution</xlabel>
</gnuplot>

<AllTests>
    <replace_lines>
      <maxTime> 20 </maxTime>
    </replace_lines>
    <replace_values>
      /Uintah_specification/Grid/BoundaryConditions/Face[@side='y+']/BCType[@id = '0' and @label ='Velocity' and @var='Dirichlet']/value : [1,0,0] 
    </replace_values>
</AllTests>
<Test>
    <Title>16</Title>
    <sus_cmd>mpirun -np 6 sus -mpi </sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -pDir 1 -mat 0 -plot true -Re 5000</postProcess_cmd>
    <x>16</x>
    <replace_lines>
      <resolution>  [16,16,1]     </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>32</Title>
    <sus_cmd>mpirun -np 6 sus -mpi</sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -pDir 1 -mat 0 -plot true -Re 5000</postProcess_cmd>
        <x>32</x>
    <replace_lines>
      <resolution>  [32,32,1]     </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>64</Title>
    <sus_cmd>mpirun -np 6 sus -mpi</sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -pDir 1 -mat 0 -plot true -Re 5000</postProcess_cmd>
    <x>64</x>
    <replace_lines>
      <resolution>  [64,64,1]     </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>128</Title>
    <sus_cmd>mpirun -np 6 sus -mpi</sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -pDir 1 -mat 0 -plot true -Re 5000</postProcess_cmd>
        <x>128</x>
    <replace_lines>
      <resolution>  [128,128,1]     </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>256</Title>
    <sus_cmd>mpirun -np 6 sus -mpi</sus_cmd>
    <postProcess_cmd>compare_cavityFlow.m -pDir 1 -mat 0 -plot true -Re 5000</postProcess_cmd>
        <x>256</x>
    <replace_lines>
      <resolution>  [256,256,1]     </resolution>
    </replace_lines>
</Test>
</start>
