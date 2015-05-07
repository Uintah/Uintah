<start>
<upsFile>ConvectiveBurning.ups</upsFile>

<!-- NOTE....... plotScript is hard coded!!!!!-->

<gnuplot>
  <script>plotScript.gp</script>
  <title>Propagation Velocity vs Time</title>
  <ylabel>Instantaneous Velocity (m/s)</ylabel>
  <xlabel>Time (s)</xlabel>
</gnuplot>

<AllTests>
</AllTests>
<Test>
    <Title>ME1e8HE1e5</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>propagation_burning.m -pDir 1 -mat 1 -cellSize 1e-6 -Xold 2</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 1e8]   </momentum>
      <heat>      [ 1e5]  </heat>
    </replace_lines>
</Test>
<!--<Test>
    <Title>ME1e8HE1e8</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>propagation_burning.m -pDir 1 -mat 1 -cellSize 1e-6 -Xold 2</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 1e8]   </momentum>
      <heat>      [ 1e8]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME1e7HE1e8</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>propagation_burning.m -pDir 1 -mat 1 -cellSize 1e-6 -Xold 2</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 1e7]   </momentum>
      <heat>      [ 1e8]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME1e7HE1e5</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
   <postProcess_cmd>propagation_burning.m -pDir 1 -mat 1 -cellSize 1e-6 -Xold 2</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 1e7]   </momentum>
      <heat>      [ 1e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME1e5HE1e5</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>propagation_burning.m -pDir 1 -mat 1 -cellSize 1e-6 -Xold 2</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 1e3]   </momentum>
      <heat>      [ 1e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME1e3HE1e8</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
   <postProcess_cmd>propagation_burning.m -pDir 1 -mat 1 -cellSize 1e-6 -Xold 2</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 1e3]   </momentum>
      <heat>      [ 1e8]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME1e5HE1e8</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>propagation_burning.m -pDir 1 -mat 1 -cellSize 1e-6 -Xold 2</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 1e5]   </momentum>
      <heat>      [ 1e8]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME1e5HE1e5</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>propagation_burning.m -pDir 1 -mat 1 -cellSize 1e-6 -Xold 2</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 1e5]   </momentum>
      <heat>      [ 1e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME1e3HE1e8</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>propagation_burning.m -pDir 1 -mat 1 -cellSize 1e-6 -Xold 2</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 1e3]   </momentum>
      <heat>      [ 1e8]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME1e3HE1e5</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>propagation_burning.m -pDir 1 -mat 1 -cellSize 1e-6 -Xold 2</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 1e3]   </momentum>
      <heat>      [ 1e5]  </heat>
    </replace_lines>
</Test>-->
</start>
