<start>
<upsFile>RMCRT_test_1L.ups</upsFile>
<gnuplot>
  <script>plotScript.gp</script>s
  <title>CPU:RMCRT order-of-accuracy \\n 1 timestep (41^3)</title>
  <ylabel>Error</ylabel>
  <xlabel># of Rays</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
    <max_Timesteps>1</max_Timesteps>
    <randomSeed> true </randomSeed>
    <resolution> [41,41,41]  </resolution>
  </replace_lines>
</AllTests>

<Test>
    <Title>2</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>2</x>
    <replace_lines>
      <nDivQRays>          2        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>4</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>4</x>
    <replace_lines>
      <nDivQRays>          4        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>8</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>8</x>
    <replace_lines>
      <nDivQRays>          8        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>16</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper  -bm 1 -L 0</postProcess_cmd>
    <x>16</x>
    <replace_lines>
      <nDivQRays>          16        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>32</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>32</x>
    <replace_lines>
      <nDivQRays>          32        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>64</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>64</x>
    <replace_lines>
      <nDivQRays>          64        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>128</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0 -plot true</postProcess_cmd>
    <x>128</x>
    <replace_lines>
      <nDivQRays>          128        </nDivQRays>
    </replace_lines>
</Test>

</start>
