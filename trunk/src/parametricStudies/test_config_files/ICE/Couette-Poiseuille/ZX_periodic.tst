<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>CouettePoiseuille/ZX_periodic.ups</upsFile>

<gnuplot>
  <script>plotScript.gp</script>
  <title>ICE: Couette Poiseuille Problem (ZX) Periodic BCs (P = 1, U = 3.4453 dpdz = 100 )</title>
  <ylabel>Error</ylabel>
  <xlabel>Resolution</xlabel>
  <label> "z+\/- Periodic BCs\\ndpdz:computed from pressFC\\nTime: 1.0s, CFL: 0.4\\nAdvection: 2nd order scheme\\nSemi-implicit pressure\\n50 cells in axial direction, 1 cell 'deep'" at graph 0.01,0.15 font "Times,10" </label>
</gnuplot>

<AllTests>
  <replace_lines>
    <maxTime>             1.0    </maxTime>
    <outputInterval>      0.25   </outputInterval>
    <knob_for_diffusion>  0.6    </knob_for_diffusion>
    <dynamic_viscosity>   1.0e-2 </dynamic_viscosity>
    <z_dir> 100 </z_dir>
  </replace_lines>
  
  <!--__________________________________
    P = -0.25    with U = 13.781  dpdx = +100
    P = 1.0      with U = 3.4453  dpdx = -100
    P = -1       with U = 3.4453  dpdx = +100 
      __________________________________-->
  
  <replace_values>
    <!-- P = -0.25 
    <entry path = "Uintah_specification/CFD/ICE/fixedPressureGradient/x_dir" value =  '100' />
    <entry path = "Uintah_specification/Grid/BoundaryConditions/Face[@side='x+']/BCType[@id = '0' and @label ='Velocity' and @var='Dirichlet']/value" value = '[0., 0., 13.781]' />
    -->
    
    <!-- P = 1 -->
    <entry path = "Uintah_specification/CFD/ICE/fixedPressureGradient/z_dir" value =  '-100' />
    <entry path = "Uintah_specification/Grid/BoundaryConditions/Face[@side='x+']/BCType[@id = '0' and @label ='Velocity' and @var='Dirichlet']/value" value = '[0., 0., 3.4453]' />  
 
    
    <!--    P = -1 
    <entry path = "Uintah_specification/CFD/ICE/fixedPressureGradient/x_dir" value =  '100' />
    <entry path = "Uintah_specification/Grid/BoundaryConditions/Face[@side='x+']/BCType[@id = '0' and @label ='Velocity' and @var='Dirichlet']/value" value = '[0., 0., 3.4453]' />
    -->
  </replace_values>
</AllTests>

<Test>
    <Title>10</Title>
    <sus_cmd> mpirun -np 2 sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 2 -sliceDir 0 -mat 0 -P 1 -periodicBCs true -plot true</postProcess_cmd>
    <x>10</x>
    <replace_lines>
      <spacing>         [0.005,0.01,0.02]  </spacing>
      <patches>      [1,1,2]               </patches>
    </replace_lines>
</Test>

<Test>
    <Title>20</Title>
    <sus_cmd> mpirun -np 2 sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 2 -sliceDir 0 -mat 0 -P 1 -periodicBCs true -plot true</postProcess_cmd>
    <x>20</x>
    <replace_lines>
      <spacing>         [0.0025,0.01,0.02]  </spacing>
      <patches>      [1,1,2]                </patches>
    </replace_lines>
</Test>

<Test>
    <Title>25</Title>
    <sus_cmd> mpirun -np 2 sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 2 -sliceDir 0 -mat 0 -P 1 -periodicBCs true -plot true</postProcess_cmd>
    <x>25</x>
    <replace_lines>
      <spacing>         [0.002,0.01,0.02]  </spacing>
      <patches>      [1,1,2]               </patches>
    </replace_lines>
</Test>

<Test>
    <Title>40</Title>
    <sus_cmd> mpirun -np 4 sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 2 -sliceDir 0 -mat 0 -P 1 -periodicBCs true -plot true</postProcess_cmd>
    <x>40</x>
    <replace_lines>
      <spacing>         [0.00125,0.01, 0.02]  </spacing>
      <patches>      [2,1,2]                  </patches>
    </replace_lines>
</Test>

<Test>
    <Title>50</Title>
    <sus_cmd> mpirun -np 4 sus </sus_cmd>
    <postProcess_cmd>compare_Couette-Poiseuille.m -pDir 2 -sliceDir 0 -mat 0 -P 1 -periodicBCs true -plot true</postProcess_cmd>
    <x>50</x>
    <replace_lines>
      <spacing>         [0.001,0.01,0.02]  </spacing>
      <patches>      [2,1,2]               </patches>
    </replace_lines>
</Test>

</start>
