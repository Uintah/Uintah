<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>impact-Al-Al_thin.ups</upsFile>
<gnuplot></gnuplot>

<!--To see the xml paths in the file execute:    xmlstarlet el -v <xmlFile>" -->

<Test>
    <Title>FM-3e8_FSTD_1e8</Title>
    <sus_cmd> nice mpirun -np 16 sus </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-3e8_FSTD_1e8</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_mean :3e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_std :1e8
    </replace_values>
</Test>
<Test>
    <Title>FM-8e8_FSTD_1e8</Title>
    <sus_cmd> nice mpirun -np 16 sus </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-8e8_FSTD_1e8</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_mean :8e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_std :1e8
    </replace_values>
</Test>
<Test>
    <Title>FM-1.3e9_FSTD_1e8</Title>
    <sus_cmd> nice mpirun -np 16 sus </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-1.3e9_FSTD_1e8</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_mean :1.3e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_std :1e8
    </replace_values>
</Test>

<!--__________________________________-->
<Test>
    <Title>FM-3e8_FSTD_5e8</Title>
    <sus_cmd> nice mpirun -np 16 sus </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-3e8_FSTD_5e8</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_mean :3e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_std :5e8
    </replace_values>
</Test>
<Test>
    <Title>FM-8e8_FSTD_5e8</Title>
    <sus_cmd> nice mpirun -np 16 sus </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-8e8_FSTD_5e8</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_mean :8e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_std :5e8
    </replace_values>
</Test>
<Test>
    <Title>FM-1.3e9_FSTD_5e8</Title>
    <sus_cmd> nice mpirun -np 16 sus </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-1.3e9_FSTD_5e8</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_mean :1.3e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_std :5e8
    </replace_values>
</Test>

<!--__________________________________-->
<Test>
    <Title>FM-3e8_FSTD_1e9</Title>
    <sus_cmd> nice mpirun -np 16 sus </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-3e8_FSTD_1e9</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_mean :3e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_std :1e9
    </replace_values>
</Test>
<Test>
    <Title>FM-8e8_FSTD_1e9</Title>
    <sus_cmd> nice mpirun -np 16 sus </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-8e8_FSTD_1e9</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_mean :8e8
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_std :1e9
    </replace_values>
</Test>
<Test>
    <Title>FM-1.3e9_FSTD_1e9</Title>
    <sus_cmd> nice mpirun -np 16 sus </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-1.3e9_FSTD_1e9</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_mean :1.3e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='Target']/constitutive_model/failure_std :1e9
    </replace_values>
</Test>

</start>
