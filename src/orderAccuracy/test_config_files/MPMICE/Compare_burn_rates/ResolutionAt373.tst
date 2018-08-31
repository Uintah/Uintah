<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>BurnRate.ups</upsFile>

<!-- NOTE....... plotScript2 is hard coded!!!!!-->

<gnuplot>
  <script>Compare_burn_rates/plotScript373.gp</script>
  <title>Burn Rates vs Pressure At 373K</title>
  <ylabel>Burn Rate (m/s)</ylabel>
  <xlabel>Pressure (Pa)</xlabel>
</gnuplot>
<AllTests>
</AllTests>
<!--<Test>
    <Title>Resolution0.25mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 -intTemp 373</postProcess_cmd>
    <x>25</x>
    <replace_values>
        <entry path = "/Uintah_specification/Grid/Level/Box/upper"      value = '[0.200, 0.00025, 0.00025]' />
        <entry path = "/Uintah_specification/Grid/Level/Box/resolution" value = '[800,1,1]' />
        <entry path = "/Uintah_specification/MaterialProperties/MPM/material/geom_object/temperature" value = '373' />
    </replace_values>
</Test>
<Test>
    <Title>Resolution1mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 -intTemp 373</postProcess_cmd>
    <x>25</x>
    <replace_values>
         <entry path = "/Uintah_specification/Grid/Level/Box/resolution" value = '[200,1,1]' />
         <entry path = "/Uintah_specification/MaterialProperties/MPM/material/geom_object/temperature" value = '373' />
    </replace_values>
</Test>-->
<Test>
    <Title>Resolution4mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 -intTemp 373</postProcess_cmd>
    <x>25</x>
     <replace_values>
        <entry path = "/Uintah_specification/Grid/Level/Box/upper"       value = '[0.200, 0.004, 0.004]' />
        <entry path = "/Uintah_specification/Grid/Level/Box/resolution"  value = '[50,1,1]' />
        <entry path = "/Uintah_specification/MaterialProperties/MPM/material/geom_object/temperature" value = '373' />
      </replace_values>
</Test>
<Test>
    <Title>Resolution8mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 -intTemp 373</postProcess_cmd>
    <x>25</x>
      <replace_values>
        <entry path = "/Uintah_specification/Grid/Level/Box/upper"         value = '[0.200, 0.008, 0.008]' />
        <entry path = "/Uintah_specification/Grid/Level/Box/resolution"    value = '[25,1,1]' />
        <entry path = "/Uintah_specification/MaterialProperties/MPM/material/geom_object/temperature" value = '373' />
      </replace_values>
</Test>
<Test>
    <Title>Resolution10mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 -intTemp 373</postProcess_cmd>
    <x>25</x>
      <replace_values>
        <entry path = "/Uintah_specification/Grid/Level/Box/upper"       value = '[0.200, 0.01, 0.01]' />
        <entry path = "/Uintah_specification/Grid/Level/Box/resolution"  value = '[20,1,1]' />
        <entry path = "/Uintah_specification/MaterialProperties/MPM/material/geom_object/temperature" value = '373' />
      </replace_values>
</Test>
<Test>
    <Title>Resolution14mm</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 -intTemp 373</postProcess_cmd>
    <x>25</x>
    <replace_values>
         <entry path = "/Uintah_specification/Grid/Level/Box/resolution" value = '[21,1,1]' />
         <entry path = "/Uintah_specification/Grid/Level/Box/upper"      value = '[0.294, 0.014, 0.014]' />
         <entry path = "/Uintah_specification/MaterialProperties/MPM/material/geom_object/box/min" value = '[0.07, 0.0, 0.0]' />
          <entry path = "/Uintah_specification/MaterialProperties/MPM/material/geom_object/temperature" value = '373' />
    </replace_values>
</Test>
</start>
