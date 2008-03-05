<start>


<upsFile>advectPS.ups</upsFile>
<Study>Res.Study</Study>
<gnuplotFile>plotScript.gp</gnuplotFile>


<Test>
    <Title>100</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_scalar</compCommand>
    <x>100</x>
    <replace_lines>
      <resolution>   [100,1,1]          </resolution>
    </replace_lines>
</Test>

<!--
<Test>
    <Title>100</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_PS.m -type triangular -slope 1 -vel 10 -min -0.5 -max 0.5 -cells \'-istart 0 0 0 -iend 100 0 0\'  -L</compCommand>
    <x>100</x>

    <replace_lines>
      <resolution>   [100,1,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>125</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_PS.m -type triangular -slope 1 -vel 10 -min -0.5 -max 0.5 -cells \'-istart 0 0 0 -iend 124 0 0\'  -L</compCommand>
    <x>125</x>
    <replace_lines>
      <resolution>   [125,1,1]          </resolution>
    </replace_lines>
</Test>
-->
</start>
