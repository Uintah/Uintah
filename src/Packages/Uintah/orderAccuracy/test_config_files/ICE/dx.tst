<start>
<upsFile>advectPS.ups</upsFile>
<Study>Res.Study</Study>
<gnuplotFile>plotScript.gp</gnuplotFile>


<Test>
    <Title>100</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_scalar -v</compCommand>
    <x>100</x>
    <replace_lines>
      <resolution>   [100,1,1]          </resolution>
    </replace_lines>
</Test>


<Test>
    <Title>200</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_scalar -v</compCommand>
    <x>200</x>
    <replace_lines>
      <resolution>   [200,1,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>400</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_scalar -v</compCommand>
    <x>400</x>
    <replace_lines>
      <resolution>   [400,1,1]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>800</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_scalar -v</compCommand>
    <x>800</x>
    <replace_lines>
      <resolution>   [800,1,1]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>1600</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_scalar -v</compCommand>
    <x>1600</x>
    <replace_lines>
      <resolution>   [1600,1,1]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>3200</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_scalar -v</compCommand>
    <x>3200</x>
    <replace_lines>
      <resolution>   [3200,1,1]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>6400</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_scalar -v</compCommand>
    <x>6400</x>
    <replace_lines>
      <resolution>   [6400,1,1]          </resolution>
    </replace_lines>
</Test>
</start>
