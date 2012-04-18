<start>
<upsFile>DeterminingBurnRate.ups</upsFile>

<!-- NOTE....... plotScript2 is hard coded!!!!!-->

<gnuplot>
  <script>plotScript423.gp</script>
  <title>Burn Rates vs Pressure At 423K</title>
  <ylabel>Burn Rate (m/s)</ylabel>
  <xlabel>Pressure (Pa)</xlabel>
</gnuplot>
<AllTests>
</AllTests>
<!--<Test>
    <Title>Resolution0.25mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 -intTemp 423</postProcess_cmd>
    <x>25</x>
    <replace_values>
          /Uintah_specification/Grid/Level/Box/upper :[0.200, 0.00025, 0.00025]
         /Uintah_specification/Grid/Level/Box/resolution :[800,1,1]
         /Uintah_specification/MaterialProperties/MPM/material/geom_object/temperature :423
    </replace_values>
</Test>-->
<Test>
    <Title>Resolution1mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 -intTemp 423</postProcess_cmd>
    <x>25</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box/resolution :[200,1,1]
         /Uintah_specification/MaterialProperties/MPM/material/geom_object/temperature :423
    </replace_values>
</Test>
<Test>
    <Title>Resolution4mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 -intTemp 423</postProcess_cmd>
    <x>25</x>
     <replace_values>
         /Uintah_specification/Grid/Level/Box/upper :[0.200, 0.004, 0.004]
         /Uintah_specification/Grid/Level/Box/resolution :[50,1,1]
         /Uintah_specification/MaterialProperties/MPM/material/geom_object/temperature :423
      </replace_values>
</Test>
<Test>
    <Title>Resolution8mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 -intTemp 423</postProcess_cmd>
    <x>25</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box/upper :[0.200, 0.008, 0.008]
         /Uintah_specification/Grid/Level/Box/resolution :[25,1,1]
         /Uintah_specification/MaterialProperties/MPM/material/geom_object/temperature :423
    </replace_values>
</Test>
<Test>
    <Title>Resolution10mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 -intTemp 423</postProcess_cmd>
    <x>25</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box/upper :[0.200, 0.01, 0.01]
         /Uintah_specification/Grid/Level/Box/resolution :[20,1,1]
         /Uintah_specification/MaterialProperties/MPM/material/geom_object/temperature :423
    </replace_values>
</Test>
<Test>
    <Title>Resolution14mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 -intTemp 423</postProcess_cmd>
    <x>25</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box/resolution :[21,1,1]
         /Uintah_specification/Grid/Level/Box/upper :[0.294, 0.014, 0.014]
         /Uintah_specification/MaterialProperties/MPM/material/geom_object/box/min : [0.07, 0.0, 0.0]
         /Uintah_specification/MaterialProperties/MPM/material/geom_object/temperature :423
    </replace_values>
</Test>
</start>
