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
</Test>
</start>
