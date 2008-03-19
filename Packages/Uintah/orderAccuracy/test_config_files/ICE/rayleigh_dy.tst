<start>
<upsFile>rayleigh.ups</upsFile>
<Study>Res.Study</Study>
<gnuplotFile>plotScript.gp</gnuplotFile>

<AllTests>
</AllTests>
<Test>
    <Title>25</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_Rayleigh.m -pDir 1 -mat 0 -plot false</compCommand>
    <x>25</x>
    <replace_lines>
      <resolution>   [10,25,1]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>50</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_Rayleigh.m -pDir 1 -mat 0 -plot false</compCommand>
    <x>50</x>
    <replace_lines>
      <resolution>   [10,50,1]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>100</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_Rayleigh.m -pDir 1 -mat 0 -plot false</compCommand>
    <x>100</x>
    <replace_lines>
      <resolution>   [10,100,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>200</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_Rayleigh.m -pDir 1 -mat 0 -plot false</compCommand>
    <x>200</x>
    <replace_lines>
      <resolution>   [10,200,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>400</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_Rayleigh.m -pDir 1 -mat 0 -plot false</compCommand>
    <x>400</x>
    <replace_lines>
      <resolution>   [10,400,1]          </resolution>
    </replace_lines>
</Test>

</start>
