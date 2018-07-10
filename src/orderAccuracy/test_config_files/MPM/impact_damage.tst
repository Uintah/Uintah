<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>impact-Al-Al_thin.ups</upsFile>
<gnuplot></gnuplot>
<exitOnCrash> false </exitOnCrash>

<!--To see the xml paths in the file execute:    xmlstarlet el -v <xmlFile>" -->
<!--
<Test>
    <Title>FM-3e9_FSTD_20</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-3e9_FSTD_20</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_mean :3e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_std :20
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_mean :3e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_std :20
    </replace_values>
</Test>
<Test>
    <Title>FM-8e9_FSTD_20</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-8e9_FSTD_20</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_mean :8e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_std :20
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_mean :8e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_std :20
    </replace_values>
</Test>
<Test>
    <Title>FM-1.3e10_FSTD_20</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-1.3e10_FSTD_20</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_mean :1.3e10
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_std :20
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_mean :1.3e10
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_std :20
    </replace_values>
</Test>
-->
<!--__________________________________
<Test>
    <Title>FM-3e8_FSTD_5</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-3e8_FSTD_5</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_mean :3e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_std :5
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_mean :3e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_std :5
    </replace_values>
</Test>
<Test>
    <Title>FM-8e8_FSTD_5</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-8e8_FSTD_5</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_mean :8e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_std :5
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_mean :8e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_std :5
    </replace_values>
</Test>
<Test>
    <Title>FM-1.3e9_FSTD_5</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-1.3e9_FSTD_5</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_mean :1.3e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_std :5
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_mean :1.3e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_std :5
    </replace_values>
</Test>
-->
<!--__________________________________
<Test>
    <Title>FM-3e8_FSTD_10</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-3e8_FSTD_10</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_mean :3e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_std :10
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_mean :3e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_std :10
    </replace_values>
</Test>
<Test>
    <Title>FM-8e8_FSTD_10</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-8e8_FSTD_10</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_mean :8e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_std :10
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_mean :8e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_std :10
    </replace_values>
</Test>
<Test>
    <Title>FM-1.3e9_FSTD_10</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-1.3e9_FSTD_10</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_mean :1.3e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_std :10
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_mean :1.3e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_std :10
    </replace_values>
</Test>
-->
<Test>
    <Title>FM-1.3e9_FSTD_5</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-1.3e9_FSTD_5</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_mean :1.3e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_std :5
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_mean :1.3e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_std :5
    </replace_values>
</Test>
<Test>
    <Title>FM-1.3e9_FSTD_10</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-1.3e9_FSTD_10</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_mean :1.3e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_std :10
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_mean :1.3e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_std :10
    </replace_values>
</Test>
<Test>
    <Title>FM-1.3e9_FSTD_20</Title>
    <sus_cmd> nice mpirun -np 1 sus -nthreads 16 </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-1.3e9_FSTD_20</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_mean :1.3e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/damage_model/failure_std :20
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_mean :1.3e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Al_Bullet']/damage_model/failure_std :20
    </replace_values>
</Test>
</start>
