<start>
<upsFile>RMCRT_ML.ups</upsFile>
<gnuplot>
  <script>plot_refineRatio.gp</script>
  <title> RMCRT: Div Q Error versus Refinement Ratio \\n Burns and Christon Benchmark \\n 1 timestep (100 Rays per cell), Random Seed, 2 Levels</title>
  <ylabel>Div Q Error (L2norm)</ylabel>
  <xlabel>Refinement Ratio</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
    <max_Timesteps>1</max_Timesteps>
    <NoOfRays>  100  </NoOfRays>
    <randomSeed> true </randomSeed>
  </replace_lines>
</AllTests>

<Test>
    <Title>1</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper  -bm 1 </postProcess_cmd>
    <x>1</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/resolution :[41,41,41]
         /Uintah_specification/Grid/Level/Box[@label=1]/resolution :[41,41,41]
    </replace_values>
</Test>

<Test>
    <Title>2</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper  -bm 1</postProcess_cmd>
    <x>2</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/resolution :[41,41,41]
         /Uintah_specification/Grid/Level/Box[@label=1]/resolution :[82,82,82]
    </replace_values>
</Test>
<Test>
    <Title>4</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper  -bm 1</postProcess_cmd>
    <x>4</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/resolution :[41,41,41]
         /Uintah_specification/Grid/Level/Box[@label=1]/resolution :[164,164,164]
    </replace_values>
</Test>
<Test>
    <Title>8</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper  -bm 1</postProcess_cmd>
    <x>8</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/resolution :[41,41,41]
         /Uintah_specification/Grid/Level/Box[@label=1]/resolution :[328,328,328]
    </replace_values>
</Test>
<!--
<Test>
    <Title>16</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper  -bm 1</postProcess_cmd>
    <x>16</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/resolution :[41,41,41]
         /Uintah_specification/Grid/Level/Box[@label=1]/resolution :[656,656,656]
    </replace_values>
</Test>
-->
</start>
