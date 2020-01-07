<?xml version="1.0" encoding="ISO-8859-1"?>
<!--______________________________________________________________________
  This tst places a virtual radiometer in each of the 8 corners and
  shoots rays diagonally across the domain.  The intensity for
  each radiometer should be identical


           2_____________6
          /|             /|
         / |            / |
        3__+__________7   |
        |  |           |  |
        |  |           |  |
        |  |           |  |
        | 0 ___________+__4
        | /            | /
        |/             |/
        1______________5

There are also tests that place the radiometer in the center of each domain
face and shoots rays to the opposite wall.


Post processing script to extract the intensity

cd ps_results/Examples/VR

#!/bin/csh -f

foreach uda ( *.uda.000 )
  echo -n $uda | sed s/RMCRT_radiometer_//g
  $f_opt/puda -varsummary -brief $uda | sed -n 21,21p | cut -d , -f2
end

______________________________________________________________________ -->
<start>

<upsFile>RMCRT_radiometer.ups</upsFile>


<AllTests>
  <replace_lines>
    <max_Timesteps>1</max_Timesteps>
    <resolution>     [50,50,50]  </resolution>
    <allowReflect>      false    </allowReflect>
    <randomSeed>        true     </randomSeed>
    <calc_frequency>    1        </calc_frequency>
    <nRays>             100000   </nRays>
    <viewAngle>         10       </viewAngle>
  </replace_lines>
</AllTests>

<Test>
    <Title>corner_0</Title>
    <sus_cmd> sus  </sus_cmd>
    <replace_values>
      <entry path = "Uintah_specification/RMCRT/Radiometer/orientation"  value ='[1, 1, 1]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMin" value ='[0.0,  0.0,  0.0  ]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMax" value ='[0.02, 0.02, 0.02 ]' />
    </replace_values>
</Test>

<Test>
    <Title>corner_1</Title>
    <sus_cmd> sus </sus_cmd>
    <replace_values>
      <entry path = "Uintah_specification/RMCRT/Radiometer/orientation"  value ='[1, 1, -1]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMin" value ='[0.0,  0.0,  0.98]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMax" value ='[0.02, 0.02, 1.0 ]' />
    </replace_values>
</Test>

<Test>
    <Title>corner_2</Title>
    <sus_cmd> sus  </sus_cmd>
    <replace_values>
      <entry path = "Uintah_specification/RMCRT/Radiometer/orientation"  value ='[1, -1, 1]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMin" value ='[0.0,  0.98, 0.0  ]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMax" value ='[0.02, 1.0,  0.02]' />
    </replace_values>
</Test>

<Test>
    <Title>corner_3</Title>
    <sus_cmd> sus  </sus_cmd>
    <replace_values>
      <entry path = "Uintah_specification/RMCRT/Radiometer/orientation"  value ='[1, -1, -1]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMin" value ='[0.0,  0.98, 0.98 ]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMax" value ='[0.02, 1.0,  1.0   ]' />
    </replace_values>
</Test>

<Test>
    <Title>corner_4</Title>
    <sus_cmd> sus  </sus_cmd>

    <replace_values>
      <entry path = "Uintah_specification/RMCRT/Radiometer/orientation"  value ='[-1, 1, 1]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMin" value ='[0.98, 0.0,  0.0  ]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMax" value ='[1.0,  0.02, 0.02]' />
    </replace_values>
</Test>

<Test>
    <Title>corner_5</Title>
    <sus_cmd> sus  </sus_cmd>

    <replace_values>
      <entry path = "Uintah_specification/RMCRT/Radiometer/orientation"  value ='[-1, 1, -1]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMin" value ='[0.98, 0.0,  0.98]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMax" value ='[1.0,  0.02, 1.0 ]' />
    </replace_values>
</Test>

<Test>
    <Title>corner_6</Title>
    <sus_cmd> sus  </sus_cmd>

    <replace_values>
      <entry path = "Uintah_specification/RMCRT/Radiometer/orientation"  value ='[-1, -1, 1]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMin" value ='[ 0.98, 0.98, 0.0  ]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMax" value ='[ 1.0,  1.0,  0.02]' />
    </replace_values>
</Test>

<Test>
    <Title>corner_7</Title>
    <sus_cmd> sus  </sus_cmd>

    <replace_values>
      <entry path = "Uintah_specification/RMCRT/Radiometer/orientation"  value ='[-1, -1, -1]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMin" value ='[0.98, 0.98, 0.98]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMax" value ='[1.0,  1.0,   1.0 ]' />
    </replace_values>
</Test>

<!--__________________________________-->
<!--  Principle axis tests  -->
<!--  x- -> x+  -->
<!--  x+ -> x-  -->
<!--  y- -> x+  -->
<!--  z- -> z+  -->
<!--  z+ -> z-  -->

<Test>
    <Title>xDir_+dir</Title>
    <sus_cmd> sus  </sus_cmd>
    <replace_values>
      <entry path = "Uintah_specification/RMCRT/Radiometer/orientation"  value ='[1,    0, 0]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMin" value ='[0.0,  0.49, 0.49 ]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMax" value ='[0.02, 0.51, 0.51 ]' />
    </replace_values>
</Test>

<Test>
    <Title>xDir_-dir</Title>
    <sus_cmd> sus  </sus_cmd>
    <replace_values>
      <entry path = "Uintah_specification/RMCRT/Radiometer/orientation"  value ='[-1,  0, 0]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMin" value ='[0.98,0.49, 0.49 ]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMax" value ='[1.0, 0.51, 0.51 ]' />
    </replace_values>
</Test>

<Test>
    <Title>yDir_+dir</Title>
    <sus_cmd> sus  </sus_cmd>
    <replace_values>
      <entry path = "Uintah_specification/RMCRT/Radiometer/orientation"  value ='[0,    1,     0]'    />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMin" value ='[0.49, 0.0,   0.49 ]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMax" value ='[0.51, 0.02,  0.51 ]' />
    </replace_values>
</Test>

<Test>
    <Title>yDir_-dir</Title>
    <sus_cmd> sus  </sus_cmd>
    <replace_values>
      <entry path = "Uintah_specification/RMCRT/Radiometer/orientation"  value ='[0,    -1,   0]'   />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMin" value ='[0.49, 0.98, 0.49 ]' />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMax" value ='[0.51, 1.0,  0.51 ]' />
    </replace_values>
</Test>

<Test>
    <Title>zDir_+dir</Title>
    <sus_cmd> sus  </sus_cmd>
    <replace_values>
      <entry path = "Uintah_specification/RMCRT/Radiometer/orientation"  value ='[0,      0,   1   ]'  />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMin" value ='[0.49,  0.49, 0.0 ]'  />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMax" value ='[0.51,  0.51, 0.02 ]' />
    </replace_values>
</Test>

<Test>
    <Title>zDir_-dir</Title>
    <sus_cmd> sus  </sus_cmd>
    <replace_values>
      <entry path = "Uintah_specification/RMCRT/Radiometer/orientation"  value ='[0,      0,   -1 ]'  />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMin" value ='[0.49,  0.49, 0.98 ]'  />
      <entry path = "Uintah_specification/RMCRT/Radiometer/locationsMax" value ='[0.51,  0.51, 1.0 ]' />
    </replace_values>
</Test>

</start>




