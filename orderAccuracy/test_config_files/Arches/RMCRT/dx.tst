<start>
<upsFile>rmcrt_test.ups</upsFile>
<gnuplot>
  <script>plotScript_dx.gp</script>s
  <title> "RMCRT Order-Of-Accuracy \n Burns & Christon Benchmark \\n 1 timestep (100 Rays per cell), Random Seed"</title>
  <ylabel>L2norm Error</ylabel>
  <xlabel>Grid Cells On Each Axis</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
    <max_Timesteps>1</max_Timesteps>
    <NoOfRays>  100  </NoOfRays>
    <randomSeed> true </randomSeed>
  </replace_lines>
</AllTests>


<Test>
    <Title>10</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper  -bm 1 -L 0</postProcess_cmd>
    <x>10</x>
    <replace_lines>
       <resolution>   [10,10,10]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>20</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>20</x>
    <replace_lines>
       <resolution>   [20,20,20]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>30</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>30</x>
    <replace_lines>
       <resolution>   [30,30,30]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>40</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>40</x>
    <replace_lines>
       <resolution>   [40,40,40]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>50</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>50</x>
    <replace_lines>
       <resolution>   [50,50,50]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>60</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>60</x>
    <replace_lines>
       <resolution>   [60,60,60]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>70</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>40</x>
    <replace_lines>
       <resolution>   [40,40,40]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>80</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>80</x>
    <replace_lines>
       <resolution>   [80,80,80]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>90</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>90</x>
    <replace_lines>
       <resolution>   [90,90,90]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>100</Title>
    <sus_cmd> mpirun -np 1 sus </sus_cmd>
    <postProcess_cmd>RMCRT_wrapper -bm 1 -L 0</postProcess_cmd>
    <x>100</x>
    <replace_lines>
       <resolution>   [100,100,100]          </resolution>
    </replace_lines>
</Test>

</start>
