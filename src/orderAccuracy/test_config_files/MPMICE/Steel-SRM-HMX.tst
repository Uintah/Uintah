<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>SE_smallCylDetSym-Steel-SRM-HMX.ups</upsFile>
<gnuplot></gnuplot>
<exitOnCrash> false </exitOnCrash>

<!--To see the xml paths in the file execute:    xmlstarlet el -v <xmlFile>" -->

<!--
<AllTests>
  <replace_lines>
    <max_Timesteps>10</max_Timesteps>
  </replace_lines>
</AllTests>
-->

<Test>
    <Title>FM-6e9_std_10</Title>
    <sus_cmd> nice mpirun -np 20 sus   </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-6e9_std_10</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_mean :6e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_std :10
    </replace_values>
</Test>
<Test>
    <Title>FM-9e9_std_10</Title>
    <sus_cmd> nice mpirun -np 20 sus   </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-9e9_std_10</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_mean :9e9
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_std :10
    </replace_values>
</Test>
<Test>
    <Title>FM-2e10_std_10</Title>
    <sus_cmd> nice mpirun -np 20 sus   </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-2e10_std_10</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_mean :2e10
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_std :10
    </replace_values>
</Test>

<Test>
    <Title>FM-5e10_std_10</Title>
    <sus_cmd> nice mpirun -np 20 sus   </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-5e10_std_10</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_mean :5e10
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_std :10
    </replace_values>
</Test>
<!--
<Test>
    <Title>FM-1e10_std_10</Title>
    <sus_cmd> nice mpirun -np 20 sus   </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-1e10_std_10</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_mean :1e10
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_std :10
    </replace_values>
</Test>
<Test>
    <Title>FM-5e10_std_10</Title>
    <sus_cmd> nice mpirun -np 20 sus   </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-5e10_std_10</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_mean :5e10
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_std :10
    </replace_values>
</Test>
<Test>
    <Title>FM-1e11_std_10</Title>
    <sus_cmd> nice mpirun -np 20 sus   </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-1e11_std_10</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_mean :1e11
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_std :10
    </replace_values>
</Test>
<Test>
    <Title>FM-5e11_std_10</Title>
    <sus_cmd> nice mpirun -np 20 sus   </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-5e11_std_10</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_mean :5e11
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_std :10
    </replace_values>
</Test>
<Test>
    <Title>FM-1e12_std_10</Title>
    <sus_cmd> nice mpirun -np 20 sus   </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>FM-1e12_std_10</x>
    <replace_values>
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_mean :1e12
         /Uintah_specification/MaterialProperties/MPM/material[@name='steel_cylinder']/damage_model/failure_std :10
    </replace_values>
</Test>
-->
</start>
