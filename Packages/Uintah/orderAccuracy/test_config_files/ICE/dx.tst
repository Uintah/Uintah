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
       <delt_init>   2.0e-5             </delt_init>
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
      <delt_init>    1.0e-5             </delt_init>
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
      <delt_init>    5.0e-6             </delt_init>
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
      <delt_init>    2.5e-6             </delt_init>
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
      <delt_init>    1.25e-6             </delt_init>
      <resolution>   [1600,1,1]          </resolution>
    </replace_lines>
</Test>

</start>
