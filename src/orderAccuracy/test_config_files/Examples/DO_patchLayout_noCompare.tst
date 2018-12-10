<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>RMCRT_bm1_DO.ups</upsFile>
<gnuplot>
  <script>plot_finePatches.gp</script>
  <title> notused</title>
  <ylabel>notused</ylabel>
  <xlabel>notused</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
    <nDivQRays>  1  </nDivQRays>
    <randomSeed> false </randomSeed>
    <halo>           [2,2,2]      </halo>
    <outputTimestepInterval>6</outputTimestepInterval>
  </replace_lines>
  <replace_values>
    <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/resolution" value ='[33,33,33]' />
    <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/resolution" value ='[66,66,66]' />
  </replace_values>
</AllTests>

<Test>
    <Title>111.311</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>111.311</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[1,1,1]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,1,1]' />
    </replace_values>
</Test>
<Test>
    <Title>311.311</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>311.311</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[3,1,1]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,1,1]' />
    </replace_values>
</Test>
<Test>
    <Title>131.311</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>131.311</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[1,3,1]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,1,1]' />
    </replace_values>
</Test>
<Test>
    <Title>113.311</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>113.311</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[1,1,3]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,1,1]' />
    </replace_values>
</Test>

<Test>
    <Title>331.311</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>331.311</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[3,3,1]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,1,1]' />
    </replace_values>
</Test>
<Test>
    <Title>313.311</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>313.311</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[3,1,3]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,1,1]' />
    </replace_values>
</Test>
<Test>
    <Title>333.311</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>333.311</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[3,3,3]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,1,1]' />
    </replace_values>
</Test>
<!--__________________________________-->

<Test>
    <Title>111.113</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>111.113</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[1,1,1]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[1,1,3]' />
    </replace_values>
</Test>
<Test>
    <Title>113.113</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>113.113</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[1,1,3]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[1,1,3]' />
    </replace_values>
</Test>
<Test>
    <Title>131.113</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>131.113</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[1,3,1]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[1,1,3]' />
    </replace_values>
</Test>
<Test>
    <Title>311.113</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>113.113</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[3,1,1]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[1,1,3]' />
    </replace_values>
</Test>

<Test>
    <Title>331.113</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>331.113</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[3,3,1]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[1,1,3]' />
    </replace_values>
</Test>
<Test>
    <Title>313.113</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>313.113</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[3,1,3]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[1,1,3]' />
    </replace_values>
</Test>
<Test>
    <Title>333.113</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>333.113</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[3,3,3]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[1,1,3]' />
    </replace_values>
</Test>

<!--__________________________________-->

<Test>
    <Title>111.313</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>111.313</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[1,1,1]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,1,3]' />
    </replace_values>
</Test>
<Test>
    <Title>311.313</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>311.313</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[3,1,1]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,1,3]' />
    </replace_values>
</Test>
<Test>
    <Title>131.313</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>131.313</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[1,3,1]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,1,3]' />
    </replace_values>
</Test>
<Test>
    <Title>113.313</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>113.313</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[1,1,3]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,1,3]' />
    </replace_values>
</Test>

<Test>
    <Title>331.313</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>331.313</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[3,3,1]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,1,3]' />
    </replace_values>
</Test>
<Test>
    <Title>313.313</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>313.313</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[3,1,3]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,1,3]' />
    </replace_values>
</Test>
<Test>
    <Title>333.313</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>333.313</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[3,3,3]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,1,3]' />
    </replace_values>
</Test>

<!--__________________________________-->

<Test>
    <Title>111.333</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>111.333</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[1,1,1]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,3,3]' />
    </replace_values>
</Test>
<Test>
    <Title>313.333</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>313.333</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[3,1,1]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,3,3]' />
    </replace_values>
</Test>
<Test>
    <Title>131.333</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>131.333</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[1,3,1]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,3,3]' />
    </replace_values>
</Test>
<Test>
    <Title>113.333</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>113.333</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[1,1,3]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,3,3]' />
    </replace_values>
</Test>

<Test>
    <Title>331.333</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>331.333</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[3,3,1]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,3,3]' />
    </replace_values>
</Test>
<Test>
    <Title>313.333</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>313.333</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[3,1,3]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,3,3]' />
    </replace_values>
</Test>
<Test>
    <Title>333.333</Title>
    <sus_cmd> mpirun -np 1 sus -nthreads 8 -gpu  </sus_cmd>
    <postProcess_cmd></postProcess_cmd>
    <x>333.333</x>
    <replace_values>
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=0]/patches" value ='[3,3,3]' />
      <entry path = "/Uintah_specification/Grid/Level/Box[@label=1]/patches" value ='[3,3,3]' />
    </replace_values>
</Test>

</start>
