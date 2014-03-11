<start>
<upsFile>advect_3L_1D.ups</upsFile>
<gnuplot>
</gnuplot>

<Test>
    <Title>1_1_1</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>pudaVarsummary -ts 5 -var p.stress</postProcess_cmd>
    <x>1</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/patches :[1,1,1]
         /Uintah_specification/Grid/Level/Box[@label=1]/patches :[1,1,1]
         /Uintah_specification/Grid/Level/Box[@label=2]/patches :[1,1,1]
    </replace_values>
</Test>

<Test>
    <Title>1_2_1</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd> pudaVarsummary -ts 5 -var p.stress  </postProcess_cmd>
    <x>2</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/patches :[1,1,1]
         /Uintah_specification/Grid/Level/Box[@label=1]/patches :[2,1,1]
         /Uintah_specification/Grid/Level/Box[@label=2]/patches :[1,1,1]
    </replace_values>
</Test>

<Test>
    <Title>3_2_1</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>  pudaVarsummary -ts 5 -var p.stress  </postProcess_cmd>
    <x>3</x>
    <replace_values>
         /Uintah_specification/Grid/Level/Box[@label=0]/patches :[3,1,1]
         /Uintah_specification/Grid/Level/Box[@label=1]/patches :[2,1,1]
         /Uintah_specification/Grid/Level/Box[@label=2]/patches :[1,1,1]
    </replace_values>
</Test>

</start>
