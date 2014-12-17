<start>
<upsFile>advect_3L_3D.ups</upsFile>
<gnuplot>
</gnuplot>

<Test>
    <Title>X_1_1_1</Title>
    <sus_cmd> mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd>pudaVarsummary -ts 5 -var p.stress</postProcess_cmd>
    <x>1</x>
    
    <replace_lines>
      <gravity>[100,0,0]</gravity>
    </replace_lines>
    
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/patches :[1,1,1]
         /Uintah_specification/Grid/Level/Box[@label=1]/patches :[1,1,1]
         /Uintah_specification/Grid/Level/Box[@label=2]/patches :[1,1,1]
    </replace_values>
</Test>

<Test>
    <Title>X_1_2_1</Title>
    <sus_cmd> mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd> pudaVarsummary -ts 5 -var p.stress  </postProcess_cmd>
    <x>2</x>
    
    <replace_lines>
      <gravity>[100,0,0]</gravity>
    </replace_lines>
    
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/patches :[1,1,1]
         /Uintah_specification/Grid/Level/Box[@label=1]/patches :[2,2,2]
         /Uintah_specification/Grid/Level/Box[@label=2]/patches :[1,1,1]
    </replace_values>
</Test>

<Test>
    <Title>X_3_2_1</Title>
    <sus_cmd> mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd>  pudaVarsummary -ts 5 -var p.stress  </postProcess_cmd>
    <x>3</x>
    
    <replace_lines>
      <gravity>[100,0,0]</gravity>
    </replace_lines>
    
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/patches :[3,3,3]
         /Uintah_specification/Grid/Level/Box[@label=1]/patches :[2,2,2]
         /Uintah_specification/Grid/Level/Box[@label=2]/patches :[1,1,1]
    </replace_values>
</Test>


<Test>
    <Title>XY_1_1_1</Title>
    <sus_cmd> mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd>pudaVarsummary -ts 5 -var p.stress</postProcess_cmd>
    <x>4</x>
    
    <replace_lines>
      <gravity>[100,100,0]</gravity>
    </replace_lines>
    
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/patches :[1,1,1]
         /Uintah_specification/Grid/Level/Box[@label=1]/patches :[1,1,1]
         /Uintah_specification/Grid/Level/Box[@label=2]/patches :[1,1,1]
    </replace_values>
</Test>

<Test>
    <Title>XY_1_2_1</Title>
    <sus_cmd> mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd> pudaVarsummary -ts 5 -var p.stress  </postProcess_cmd>
    <x>5</x>
    
    <replace_lines>
      <gravity>[100,100,0]</gravity>
    </replace_lines>
    
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/patches :[1,1,1]
         /Uintah_specification/Grid/Level/Box[@label=1]/patches :[2,2,2]
         /Uintah_specification/Grid/Level/Box[@label=2]/patches :[1,1,1]
    </replace_values>
</Test>

<Test>
    <Title>XY_3_2_1</Title>
    <sus_cmd> mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd>  pudaVarsummary -ts 5 -var p.stress  </postProcess_cmd>
    <x>6</x>
    
    <replace_lines>
      <gravity>[100,100,0]</gravity>
    </replace_lines>
    
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/patches :[3,3,3]
         /Uintah_specification/Grid/Level/Box[@label=1]/patches :[2,2,2]
         /Uintah_specification/Grid/Level/Box[@label=2]/patches :[1,1,1]
    </replace_values>
</Test>


<Test>
    <Title>XYZ_1_1_1</Title>
    <sus_cmd> mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd>pudaVarsummary -ts 5 -var p.stress</postProcess_cmd>
    <x>7</x>
    
    <replace_lines>
      <gravity>[100,100,100]</gravity>
    </replace_lines>
    
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/patches :[1,1,1]
         /Uintah_specification/Grid/Level/Box[@label=1]/patches :[1,1,1]
         /Uintah_specification/Grid/Level/Box[@label=2]/patches :[1,1,1]
    </replace_values>
</Test>

<Test>
    <Title>XYZ_1_2_1</Title>
    <sus_cmd> mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd> pudaVarsummary -ts 5 -var p.stress  </postProcess_cmd>
    <x>8</x>
    
    <replace_lines>
      <gravity>[100,100,0]</gravity>
    </replace_lines>
    
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/patches :[1,1,1]
         /Uintah_specification/Grid/Level/Box[@label=1]/patches :[2,2,2]
         /Uintah_specification/Grid/Level/Box[@label=2]/patches :[1,1,1]
    </replace_values>
</Test>

<Test>
    <Title>XYZ_3_2_1</Title>
    <sus_cmd> mpirun -np 8 sus </sus_cmd>
    <postProcess_cmd>  pudaVarsummary -ts 5 -var p.stress  </postProcess_cmd>
    <x>9</x>
    
    <replace_lines>
      <gravity>[100,100,0]</gravity>
    </replace_lines>
    
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/patches :[3,3,3]
         /Uintah_specification/Grid/Level/Box[@label=1]/patches :[2,2,2]
         /Uintah_specification/Grid/Level/Box[@label=2]/patches :[1,1,1]
    </replace_values>
</Test>



</start>
