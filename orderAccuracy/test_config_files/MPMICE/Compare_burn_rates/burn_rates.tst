<start>
<upsFile>DeterminingBurnRate.ups</upsFile>

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
    <Title>ME1e8HE1e5</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 1e8]   </momentum>
      <heat>      [ 1e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME1e8HE3e5</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 1e8]   </momentum>
      <heat>      [ 3e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME1e8HE0.8e5</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 1e8]   </momentum>
      <heat>      [ 0.8e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME3e8HE1e5</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 3e8]   </momentum>
      <heat>      [ 1e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME3e8HE3e5</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 3e8]   </momentum>
      <heat>      [ 3e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME3e8HE0.8e5</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 3e8]   </momentum>
      <heat>      [ 0.8e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME0.8e8HE1e5</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 0.8e8]   </momentum>
      <heat>      [ 1e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME0.8e8HE3e5</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 0.8e8]   </momentum>
      <heat>      [ 3e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>ME0.8e8H0.8e5</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <momentum>  [ 0.8e8]   </momentum>
      <heat>      [ 0.8e5]  </heat>
    </replace_lines>
</Test>
<Test>
    <Title>Resolution1mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box/resolution :[100,1,1]
    </replace_values>
</Test>
<Test>
    <Title>Resolution1.33mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
     <replace_values>
         /Uintah_specification/Grid/Level/Box/resolution :[75,1,1]
    </replace_values>
</Test>
<Test>
    <Title>Resolution2mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box/resolution :[50,1,1]
    </replace_values>
</Test>
<Test>
    <Title>Resolution0.5mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box/resolution :[200,1,1]
    </replace_values>
</Test>
<Test>
    <Title>Resolution0.25mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box/resolution :[400,1,1]
    </replace_values>
</Test>
<Test>
    <Title>8ParticlesPerCell</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material/geom_object/res :[2,2,2]
    </replace_values>
</Test>
<Test>
    <Title>27ParticlesPerCell</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material/geom_object/res :[3,3,3]
    </replace_values>
</Test>
<Test>
    <Title>1ParticlePerCell</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material/geom_object/res :[1,1,1]
    </replace_values>
</Test>-->
</start>
