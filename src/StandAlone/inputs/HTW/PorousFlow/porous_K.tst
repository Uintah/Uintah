<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>porousLODI.ups</upsFile>


<!--
<susTimeout_minutes> 10 </susTimeout_minutes>
-->
<exitOnCrash> false </exitOnCrash>
<!--
<AllTests>

  <replace_lines>
    <max_Timesteps>       1000 </max_Timesteps>
  </replace_lines>

</AllTests>
-->

<Test>
    <Title>1e12</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>processMassFlowRate</postProcess_cmd>
   
    <x>1e12</x>
    <replace_lines>
            <momentum>       [1e12]     </momentum>
    </replace_lines>
</Test>

<Test>
    <Title>1e9</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>processMassFlowRate</postProcess_cmd>
    <x>1e9</x>
    <replace_lines>
            <momentum>       [1e9]     </momentum>
    </replace_lines>
</Test>
<Test>
    <Title>1e8</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>processMassFlowRate</postProcess_cmd>
    <x>1e8</x>
    <replace_lines>
            <momentum>       [1e8]     </momentum>
    </replace_lines>
</Test>
<Test>
    <Title>1e7</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>processMassFlowRate</postProcess_cmd>
    <x>1e7</x>
    <replace_lines>
            <momentum>       [1e7]     </momentum>
    </replace_lines>
</Test>
<Test>
    <Title>1e6</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>processMassFlowRate</postProcess_cmd>
    <x>1e6</x>
    <replace_lines>
       <momentum>       [1e6]     </momentum>
    </replace_lines>
</Test>
<Test>
    <Title>1e5</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>processMassFlowRate</postProcess_cmd>
    <x>1e5</x>
    <replace_lines>
            <momentum>       [1e5]     </momentum>
    </replace_lines>
</Test>
<Test>
    <Title>1e4</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>processMassFlowRate</postProcess_cmd>
    <x>1e4</x>
    <replace_lines>
            <momentum>       [1e4]     </momentum>
    </replace_lines>
</Test>
<Test>
    <Title>1e3</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>processMassFlowRate</postProcess_cmd>
    <x>1e3</x>
    <replace_lines>
            <momentum>       [1e3]     </momentum>
    </replace_lines>
</Test>
<Test>
    <Title>1e2</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>processMassFlowRate</postProcess_cmd>
    <x>1e2</x>
    <replace_lines>
            <momentum>       [1e2]     </momentum>
    </replace_lines>
</Test>
<Test>
    <Title>1e1</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>processMassFlowRate</postProcess_cmd>
    <x>1e1</x>
    <replace_lines>
            <momentum>       [1e1]     </momentum>
    </replace_lines>
</Test>
<Test>
    <Title>1e0</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>processMassFlowRate</postProcess_cmd>
    <x>1e0</x>
    <replace_lines>
            <momentum>       [1e0]     </momentum>
    </replace_lines>
</Test>

</start>
