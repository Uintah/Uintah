<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>porous.ups</upsFile>
<gnuplotFile>plotScript.gp</gnuplotFile>

<!--
<susTimeout_minutes> 10 </susTimeout_minutes>
-->
<exitOnCrash> false </exitOnCrash>

<AllTests>
</AllTests>
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
<!--
<Test>
    <Title>100</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -pDir 1 -mat 1 -plot false</postProcess_cmd>
    <x>100</x>
    <replace_lines>
      <resolution>   [10,100,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>200</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -pDir 1 -mat 1 -plot false</postProcess_cmd>
    <x>200</x>
    <replace_lines>
      <resolution>   [10,200,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>400</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -pDir 1 -mat 1 -plot false</postProcess_cmd>
    <x>400</x>
    <replace_lines>
      <resolution>   [10,400,1]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>800</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>compare_Rayleigh.m -pDir 1 -mat 1 -plot false</postProcess_cmd>
    <x>800</x>
    <replace_lines>
      <resolution>   [10,800,1]          </resolution>
    </replace_lines>
</Test>
-->
</start>
