<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>BurnRate.ups</upsFile>

<!-- NOTE....... plotScript2 is hard coded!!!!!-->

<gnuplot>
  <script>Compare_burn_rates/plotScript.gp</script>
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
          <entry path = "/Uintah_specification/MaterialProperties/MPM/material/geom_object/volumeFraction" value ='1.0' />
          <entry path = "/Uintah_specification/MaterialProperties/ICE/material/geom_object/volumeFraction" value ='0.0' />
    </replace_values>
</Test>
<Test>
    <Title>VolumeFraction0.9</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 </postProcess_cmd>
    <x>25</x>
    <replace_values>
          <entry path = "/Uintah_specification/MaterialProperties/MPM/material/geom_object/volumeFraction" value ='0.9' />
          <entry path = "/Uintah_specification/MaterialProperties/ICE/material/geom_object/volumeFraction" value ='0.1' />
    </replace_values>
</Test>
<Test>
    <Title>VolumeFraction0.8</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 </postProcess_cmd>
    <x>25</x>
    <replace_values>
          <entry path = "/Uintah_specification/MaterialProperties/MPM/material/geom_object/volumeFraction" value ='0.8' />
          <entry path = "/Uintah_specification/MaterialProperties/ICE/material/geom_object/volumeFraction" value ='0.2' />
    </replace_values>
</Test>
<Test>
    <Title>VolumeFraction0.7</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 </postProcess_cmd>
    <x>25</x>
    <replace_values>
          <entry path = "/Uintah_specification/MaterialProperties/MPM/material/geom_object/volumeFraction" value ='0.7' />
          <entry path = "/Uintah_specification/MaterialProperties/ICE/material/geom_object/volumeFraction" value ='0.3' />
    </replace_values>
</Test>
<Test>
    <Title>VolumeFraction0.6</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832 </postProcess_cmd>
    <x>25</x>
    <replace_values>
          <entry path = "/Uintah_specification/MaterialProperties/MPM/material/geom_object/volumeFraction" value ='0.6' />
          <entry path = "/Uintah_specification/MaterialProperties/ICE/material/geom_object/volumeFraction" value ='0.4' />
    </replace_values>
</Test>
</start>
