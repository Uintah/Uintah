<?xml version="1.0" encoding="ISO-8859-1"?>

<!--______________________________________________________________________
#
# Parametric study to reproduce the results in the reference.  
# The domain length was computed using the script:
# cavityFlowParameters.m
#
#  Note you have to let the problems run for a long time for them to reach steady state.
# Reference:'
#   Varakos, G., Mitsoulis, E., and Assimacopoulos, D.
#   Natural Convection Flow In A Square Cavity Revisited: Laminar and Turbulent Models With Wall Functions
#   International Journal for Numerical Methods in Fluids, Vol 18, 695-719, 1994
#
#  Add:
# whatToRun.xml:  <postProcessCmd_path>/home/harman/UintahProjects/NaturalConvection_validation/</postProcessCmd_path>
#
# You'll want to only output the last timestep
#
#  <outputLastTimestep> true </outputLastTimestep>
______________________________________________________________________-->


<start>
<upsFile>naturalConvectionCavity2mat_dx.ups</upsFile>

<AllTests>
</AllTests>

<Test>
    <Title>Ra_1e3_96</Title>
    <sus_cmd>mpirun -np 9 sus </sus_cmd>
    <postProcess_cmd>postProcess.m -mat 0 -Ra '1e3' </postProcess_cmd>
    <postProcess_cmd>postProcess.m -mat 1 -Ra '1e3' </postProcess_cmd>
    <x>1e3</x>
    <replace_lines>
      <maxTime>            2     </maxTime>
      <outputInterval>    0.2    </outputInterval>
      <upper>         [8.411751e-03, 8.411751e-03, 8.762240e-05]   </upper>
      <patches>      [3,3,1]     </patches>
      <resolution>   [96,96,1]   </resolution>
    </replace_lines>
    <replace_values>
      <entry path = "Uintah_specification/MaterialProperties/ICE/material/geom_object/box[@label='box']/min" value = '[2.803910e-02, 2.803910e-02, 0.0]' />
      <entry path = "Uintah_specification/MaterialProperties/ICE/material/geom_object/box[@label='box']/max" value = '[5.607821e-02, 5.607821e-02, 2.0]' />
    </replace_values>
</Test>

<Test>
    <Title>Ra_1e4_96</Title>
    <sus_cmd>mpirun -np 9 sus </sus_cmd>
    <postProcess_cmd>postProcess.m -mat 0 -Ra '1e4'</postProcess_cmd>
    <postProcess_cmd>postProcess.m -mat 1 -Ra '1e4'</postProcess_cmd>
    <x>1e4</x>
    <replace_lines>
      <maxTime>            4     </maxTime>
      <outputInterval>    0.4   </outputInterval>
      <upper>         [1.812255e-02, 1.812255e-02, 1.887766e-04]   </upper>
      <patches>      [3,3,1]     </patches>
      <resolution>   [96,96,1]   </resolution>
    </replace_lines>
    <replace_values>
      <entry path = "Uintah_specification/MaterialProperties/ICE/material/geom_object/box[@label='box']/min" value = '[6.040851e-03, 6.040851e-03, 0.000000e+00]' />
      <entry path = "Uintah_specification/MaterialProperties/ICE/material/geom_object/box[@label='box']/max" value = '[1.208170e-02, 1.208170e-02, 2.000000e+00]' />
    </replace_values>
</Test>

<Test>
    <Title>Ra_1e5_96</Title>
    <sus_cmd>mpirun -np 9 sus </sus_cmd>
    <postProcess_cmd>postProcess.m -mat 0 -Ra '1e5'</postProcess_cmd>
    <postProcess_cmd>postProcess.m -mat 1 -Ra '1e5'</postProcess_cmd>
    <x>1e5</x>
    <replace_lines>
      <maxTime>            25     </maxTime>
      <outputInterval>     2.5   </outputInterval>
      <upper>         [3.904383e-02, 3.904383e-02, 4.067065e-04]   </upper>
      <patches>      [3,3,1]      </patches>
      <resolution>   [96,96,1]    </resolution>
    </replace_lines>
    <replace_values>
      <entry path = "Uintah_specification/MaterialProperties/ICE/material/geom_object/box[@label='box']/min" value = '[1.301461e-02, 1.301461e-02, 0.000000e+00]' />
      <entry path = "Uintah_specification/MaterialProperties/ICE/material/geom_object/box[@label='box']/max" value = '[2.602922e-02, 2.602922e-02, 2.000000e+00]' />
    </replace_values>
</Test>

<Test>
    <Title>Ra_1e6_96</Title>
    <sus_cmd>mpirun -np 9 sus </sus_cmd>
    <postProcess_cmd>postProcess.m -mat 0 -Ra '1e6'</postProcess_cmd>
    <postProcess_cmd>postProcess.m -mat 1 -Ra '1e6'</postProcess_cmd>
    <x>1e6</x>
    <replace_lines>
      <maxTime>            50     </maxTime>
      <outputInterval>    5.0     </outputInterval>
      <upper>         [8.411731e-02, 8.411731e-02, 8.762220e-04]  </upper>
      <patches>      [3,3,1]      </patches>
      <resolution>   [96,96,1]    </resolution>
    </replace_lines>
    <replace_values>
      <entry path = "Uintah_specification/MaterialProperties/ICE/material/geom_object/box[@label='box']/min" value = '[2.803910e-02, 2.803910e-02, 0.000000e+00]' />
      <entry path = "Uintah_specification/MaterialProperties/ICE/material/geom_object/box[@label='box']/max" value = '[5.607821e-02, 5.607821e-02, 2.000000e+00]' />
    </replace_values>
</Test>

<Test>
    <Title>Ra_1e8_96</Title>
    <sus_cmd>mpirun -np 9 sus </sus_cmd>
    <postProcess_cmd>postProcess.m -mat 0 -Ra '1e8'</postProcess_cmd>
    <postProcess_cmd>postProcess.m -mat 1 -Ra '1e8'</postProcess_cmd>
    <x>1e8</x>
    <replace_lines>
      <maxTime>            150     </maxTime>
      <outputInterval>     15      </outputInterval>
      <upper>        [3.904374e-01, 3.904374e-01, 4.067056e-03]  </upper>
      <patches>      [3,3,1]      </patches>
      <resolution>   [96,96,1]    </resolution>
    </replace_lines>
    <replace_values>
      <entry path = "Uintah_specification/MaterialProperties/ICE/material/geom_object/box[@label='box']/min" value = '[1.301458e-01, 1.301458e-01, 0.000000e+00]' />
      <entry path = "Uintah_specification/MaterialProperties/ICE/material/geom_object/box[@label='box']/max" value = '[2.602916e-01, 2.602916e-01, 2.000000e+00]' />
    </replace_values>
</Test>

<Test>
    <Title>Ra_1e10_128</Title>
    <sus_cmd>mpirun -np 16 sus </sus_cmd>
    <postProcess_cmd>postProcess.m -mat 0 -Ra '1e10'</postProcess_cmd>
    <postProcess_cmd>postProcess.m -mat 1 -Ra '1e10'</postProcess_cmd>
    <x>1e10</x>
    <replace_lines>
      <maxTime>            500    </maxTime>
      <outputInterval>     50     </outputInterval>
      <upper>        [1.812247e+00, 1.812247e+00, 1.887757e-02]  </upper>
      <patches>      [4,4,1]      </patches>
      <resolution>   [128,128,1]  </resolution>
    </replace_lines>
    <replace_values>
      <entry path = "Uintah_specification/MaterialProperties/ICE/material/geom_object/box[@label='box']/min" value = '[6.040823e-01, 6.040823e-01, 0.000000e+00]' />
      <entry path = "Uintah_specification/MaterialProperties/ICE/material/geom_object/box[@label='box']/max" value = '[1.208165e+00, 1.208165e+00, 2.000000e+00]' />
    </replace_values>
</Test>
<!--
-->
</start>
