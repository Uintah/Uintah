<start>
<upsFile>advectPS.ups</upsFile>
<Study>Res.Study</Study>
<gnuplotFile>plotScript.gp</gnuplotFile>

<AllTests>
  <replace_lines>
     <lower>        [-0.05,-0.05,-1.0]  </lower>
     <upper>        [ 0.05, 0.05, 1.0] </upper>
     <extraCells>   [1,1,0]             </extraCells>
     <periodic>     [0,0,1]             </periodic>
     <velocity>     [0,0,10.]          </velocity>
     <coeff>        [0,0,20]            </coeff>
  </replace_lines>
  <substitutions>
    <text find="z-" replace="x-" />
    <text find="z+" replace="x+" />
  </substitutions>
</AllTests>
<!--__________________________________-->
<Test>
    <Title>100</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_scalar -v</compCommand>
    <x>100</x>
    <replace_lines>
      <resolution>   [1,1,100]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>200</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_scalar -v</compCommand>
    <x>200</x>
    <replace_lines>
      <resolution>   [1,1,200]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>400</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_scalar -v</compCommand>
    <x>400</x>
    <replace_lines>
      <resolution>   [1,1,400]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>800</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_scalar -v</compCommand>
    <x>800</x>
    <replace_lines>
      <resolution>   [1,1,800]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>1600</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_scalar -v</compCommand>
    <x>1600</x>
    <replace_lines>
      <resolution>   [1,1,1600]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>3200</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_scalar -v</compCommand>
    <x>3200</x>
    <replace_lines>
      <resolution>   [1,1,3200]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>6400</Title>
    <Interactive>sus -ice </Interactive>
    <Study>Res.Study</Study>
    <compCommand>compare_scalar -v</compCommand>
    <x>6400</x>
    <replace_lines>
      <resolution>   [1,1,6400]          </resolution>
    </replace_lines>
</Test>

</start>
