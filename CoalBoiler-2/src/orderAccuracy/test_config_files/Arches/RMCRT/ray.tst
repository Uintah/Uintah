<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>rmcrt_bm1_1L.ups</upsFile>
<gnuplot>
  <script>plotScript.gp</script>s
  <title>RMCRT order-of-accuracy \\n \\n Burns and Christon Benchmark \\n1 timestep (41^3)</title>
  <ylabel>Error</ylabel>
  <xlabel># of Rays</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
    <max_Timesteps>1</max_Timesteps>
    <randomSeed> true </randomSeed>
  </replace_lines>
</AllTests>

<Test>
    <Title>8</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>8</x>
    <replace_lines>
      <nDivQRays>          8        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>16</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper  -bm 1 -L 0</postProcess_cmd>
    <x>16</x>
    <replace_lines>
      <nDivQRays>          16        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>32</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>32</x>
    <replace_lines>
      <nDivQRays>          32        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>64</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>64</x>
    <replace_lines>
      <nDivQRays>          64        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>128</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>128</x>
    <replace_lines>
      <nDivQRays>          128        </nDivQRays>
    </replace_lines>
</Test>
<!--
<Test>
    <Title>256</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>256</x>
    <replace_lines>
      <nDivQRays>          256        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>512</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>512</x>
    <replace_lines>
      <nDivQRays>          512        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>1024</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>1024</x>
    <replace_lines>
      <nDivQRays>          1024        </nDivQRays>
    </replace_lines>
</Test>
<Test>
    <Title>2048</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>2048</x>
    <replace_lines>
      <nDivQRays>          2048        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>4096</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>4096</x>
    <replace_lines>
      <nDivQRays>          4096        </nDivQRays>
    </replace_lines>
</Test>
<Test>
    <Title>8192</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>8192</x>
    <replace_lines>
      <nDivQRays>          8192        </nDivQRays>
    </replace_lines>
</Test>
-->
</start>
