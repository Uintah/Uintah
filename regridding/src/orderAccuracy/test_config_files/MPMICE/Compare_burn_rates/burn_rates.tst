<start>
<upsFile>BurnRate.ups</upsFile>

<!-- NOTE....... plotScript2 is hard coded!!!!!-->

<gnuplot>
  <script>plotScript2.gp</script>
  <title>Burn Rates vs Pressure</title>
  <ylabel>Burn Rate (m/s)</ylabel>
  <xlabel>Pressure (Pa)</xlabel>
</gnuplot>
<AllTests>
</AllTests>
<Test>
    <Title>ME3e8</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 3e8]   </momentum>
      <heat>      [ 1e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME4e8</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 4e8]   </momentum>
      <heat>      [ 1e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME5e8</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 5e8]   </momentum>
      <heat>      [ 1e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME6e8HE1e5</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 6e8]   </momentum>
      <heat>      [ 1e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME7e8</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 7e8]   </momentum>
      <heat>      [ 1e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME8e8</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 8e8]   </momentum>
      <heat>      [ 1e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME9e8</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 9e8]   </momentum>
      <heat>      [ 1e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME1e9</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 1e9]   </momentum>
      <heat>      [ 1e5]  </heat>
    </replace_lines>
</Test>
</start>
