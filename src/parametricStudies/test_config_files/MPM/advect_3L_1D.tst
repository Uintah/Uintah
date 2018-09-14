<?xml version="1.0" encoding="ISO-8859-1"?>
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
         <entry path="/Uintah_specification/Grid/Level/Box[@label=0]' />/patches" value = '[1,1,1]' />
         <entry path="/Uintah_specification/Grid/Level/Box[@label=1]' />/patches" value = '[1,1,1]' />
         <entry path="/Uintah_specification/Grid/Level/Box[@label=2]' />/patches" value = '[1,1,1]' />
    </replace_values>
</Test>

<Test>
    <Title>1_2_1</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd> pudaVarsummary -ts 5 -var p.stress  </postProcess_cmd>
    <x>2</x>
    <replace_values>
         <entry path="/Uintah_specification/Grid/Level/Box[@label=0]' />/patches" value = '[1,1,1]' />
         <entry path="/Uintah_specification/Grid/Level/Box[@label=1]' />/patches" value = '[2,1,1]' />
         <entry path="/Uintah_specification/Grid/Level/Box[@label=2]' />/patches" value = '[1,1,1]' />
    </replace_values>
</Test>

<Test>
    <Title>3_2_1</Title>
    <sus_cmd> sus </sus_cmd>
    <postProcess_cmd>  pudaVarsummary -ts 5 -var p.stress  </postProcess_cmd>
    <x>3</x>
    <replace_values>
         <entry path="/Uintah_specification/Grid/Level/Box[@label=0]' />/patches" value = '[3,1,1]' />
         <entry path="/Uintah_specification/Grid/Level/Box[@label=1]' />/patches" value = '[2,1,1]' />
         <entry path="/Uintah_specification/Grid/Level/Box[@label=2]' />/patches" value = '[1,1,1]' />
    </replace_values>
</Test>

</start>
