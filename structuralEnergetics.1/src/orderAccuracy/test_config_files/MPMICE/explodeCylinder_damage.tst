<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>SE_cylinderDet.ups</upsFile>
<gnuplot></gnuplot>
<exitOnCrash> false </exitOnCrash>

<!--To see the xml paths in the file execute:    xmlstarlet el -v <xmlFile>" -->
<!--
<Test>
    <Title>FM-5e8_FSTD_2.5</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-5e8_FSTD_2.5</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_mean :5e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_std :2.5
    </replace_values>
</Test>
<Test>
    <Title>FM-7.5e8_FSTD_2.5</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-7.5e8_FSTD_2.5</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_mean :7.5e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_std :2.5
    </replace_values>
</Test>

<Test>
    <Title>FM-1e9_FSTD_2.5</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-1e9_FSTD_2.5</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_mean :1e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_std :2.5
    </replace_values>
</Test>
-->
<!--__________________________________-->
<!--
<Test>
    <Title>FM-5e8_FSTD_5</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-5e8_FSTD_5</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_mean :5e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_std :5
    </replace_values>
</Test>

<Test>
    <Title>FM-1e9_FSTD_5</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-1e9_FSTD_5</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_mean :1e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_std :5
    </replace_values>
</Test>
<Test>
    <Title>FM-7.5e8_FSTD_5</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-7.5e8_FSTD_5</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_mean :7.5e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_std :5
    </replace_values>
</Test>
-->
<!--__________________________________-->
<Test>
    <Title>FM-5e8_FSTD_10</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-5e8_FSTD_10</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_mean :5e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_std :10
    </replace_values>
</Test>
<Test>
    <Title>FM-7.5e8_FSTD_10</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-7.5e8_FSTD_10</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_mean :7.5e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_std :10
    </replace_values>
</Test>
<Test>
    <Title>FM-1e9_FSTD_10</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-1e9_FSTD_10</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_mean :1e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='cylinder']/damage_model/failure_std :10
    </replace_values>
</Test>

</start>
