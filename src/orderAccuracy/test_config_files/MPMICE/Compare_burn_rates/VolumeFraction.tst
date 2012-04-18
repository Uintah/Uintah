<start>
<upsFile>DeterminingBurnRate.ups</upsFile>

<!-- NOTE....... plotScript2 is hard coded!!!!!-->

<gnuplot>
  <script>plotScript.gp</script>
  <title>Burn Rates vs Pressure At 298K</title>
  <ylabel>Burn Rate (m/s)</ylabel>
  <xlabel>Pressure (Pa)</xlabel>
</gnuplot>
<AllTests>
</AllTests>
<Test>
    <Title>VolumeFraction1</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 </postProcess_cmd>
    <x>25</x>
    <replace_values>
          /Uintah_specification/MaterialProperties/MPM/material/geom_object/volumeFraction :1.0
          /Uintah_specification/MaterialProperties/ICE/material/geom_object/volumeFraction :0.0
    </replace_values>
</Test>
<Test>
    <Title>VolumeFraction0.9</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 </postProcess_cmd>
    <x>25</x>
    <replace_values>
          /Uintah_specification/MaterialProperties/MPM/material/geom_object/volumeFraction :0.9
          /Uintah_specification/MaterialProperties/ICE/material/geom_object/volumeFraction :0.1
    </replace_values>
</Test>
<Test>
    <Title>VolumeFraction0.8</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 </postProcess_cmd>
    <x>25</x>
    <replace_values>
          /Uintah_specification/MaterialProperties/MPM/material/geom_object/volumeFraction :0.8
          /Uintah_specification/MaterialProperties/ICE/material/geom_object/volumeFraction :0.2
    </replace_values>
</Test>
<Test>
    <Title>VolumeFraction0.7</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 </postProcess_cmd>
    <x>25</x>
    <replace_values>
          /Uintah_specification/MaterialProperties/MPM/material/geom_object/volumeFraction :0.7
          /Uintah_specification/MaterialProperties/ICE/material/geom_object/volumeFraction :0.3
    </replace_values>
</Test>
<Test>
    <Title>VolumeFraction0.6</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 </postProcess_cmd>
    <x>25</x>
    <replace_values>
          /Uintah_specification/MaterialProperties/MPM/material/geom_object/volumeFraction :0.6
          /Uintah_specification/MaterialProperties/ICE/material/geom_object/volumeFraction :0.4
    </replace_values>
</Test>
</start>
