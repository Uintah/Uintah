<start>
<upsFile>BurnRate.ups</upsFile>

<!-- NOTE....... plotScript2 is hard coded!!!!!-->

<AllTests>
</AllTests>
<Test>
    <Title>Temp298</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
     <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material/geom_object/temperature :298
    </replace_values>
</Test>
<Test>
    <Title>Temp300</Title>
    <sus_cmd>mpirun -np 1 ./sus -svnDiff -svnStat </sus_cmd>
    <postProcess_cmd>compare_burn_rates.m -pDir 1 -mat 1 -rho_CC 1832</postProcess_cmd>
    <x>25</x>
     <replace_values>
           /Uintah_specification/MaterialProperties/MPM/material/geom_object/temperature :300
    </replace_values>
</Test>
</start>
