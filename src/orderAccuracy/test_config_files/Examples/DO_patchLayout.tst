<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>RMCRT_bm1_DO.ups</upsFile>
<gnuplot>
  <script>plot_finePatches.gp</script>
  <title> notused</title>
  <ylabel>notused</ylabel>
  <xlabel>notused</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
    <max_Timesteps>1</max_Timesteps>
    <nDivQRays>  100  </nDivQRays>
    <randomSeed> true </randomSeed>
    <halo>           [1,1,1]      </halo>
  </replace_lines>
  <replace_values>
    /Uintah_specification/Grid/Level/Box[@label=0]/resolution :[40,40,40]
    /Uintah_specification/Grid/Level/Box[@label=1]/resolution :[80,80,80]
  </replace_values>
</AllTests>

<Test>
    <Title>8</Title>
    <sus_cmd> mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper  -bm 1 -L 1</postProcess_cmd>
    <x>8</x>
    <replace_values>
      /Uintah_specification/Grid/Level/Box[@label=1]/patches :[2,2,2]
    </replace_values>
</Test>
<Test>
    <Title>16</Title>
    <sus_cmd> mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper  -bm 1 -L 1</postProcess_cmd>
    <x>16</x>
    <replace_values>
      /Uintah_specification/Grid/Level/Box[@label=1]/patches :[2,4,2]
    </replace_values>
</Test>
<Test>
    <Title>32</Title>
    <sus_cmd> mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper  -bm 1 -L 1</postProcess_cmd>
    <x>32</x>
    <replace_values>
      /Uintah_specification/Grid/Level/Box[@label=1]/patches :[4,4,2]
    </replace_values>
</Test>
<Test>
    <Title>64</Title>
    <sus_cmd> mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper  -bm 1 -L 1</postProcess_cmd>
    <x>64</x>
    <replace_values>
      /Uintah_specification/Grid/Level/Box[@label=1]/patches :[4,4,4]
    </replace_values>
</Test>
<Test>
    <Title>128</Title>
    <sus_cmd> mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper  -bm 1 -L 1</postProcess_cmd>
    <x>128</x>
    <replace_values>
      /Uintah_specification/Grid/Level/Box[@label=1]/patches :[4,8,4]
    </replace_values>
</Test>
<Test>
    <Title>256</Title>
    <sus_cmd> mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper  -bm 1 -L 1</postProcess_cmd>
    <x>256</x>
    <replace_values>
      /Uintah_specification/Grid/Level/Box[@label=1]/patches :[8,8,4]
    </replace_values>
</Test>
<Test>
    <Title>512</Title>
    <sus_cmd> mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper  -bm 1 -L 1</postProcess_cmd>
    <x>512</x>
    <replace_values>
      /Uintah_specification/Grid/Level/Box[@label=1]/patches :[8,8,8]
    </replace_values>
</Test>

</start>
