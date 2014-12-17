<start>
<upsFile>almgrenMMS.ups</upsFile>
<gnuplot>
  <script>plotScript_dt.gp</script>s
  <title>Arches:MMS:almgren:Temporal order-of-accuracy \\n dx = 32^3</title>
  <ylabel>Error</ylabel>
  <xlabel>Timestep size [s]</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
   <resolution> [32,32,32] </resolution>
   <timestep_multiplier>1</timestep_multiplier>
  </replace_lines>
</AllTests>
<Test>
    <Title>1</Title>
    <sus_cmd>mpirun -np 1 sus -mpi</sus_cmd>
    <postProcess_cmd>arches_mms.m </postProcess_cmd>
    <x>1e-5</x>
    <replace_lines>
     <delt_init>1e-5</delt_init>
    </replace_lines>
</Test>

<Test>
    <Title>2</Title>
    <sus_cmd>mpirun -np 1 sus -mpi</sus_cmd>
    <postProcess_cmd>arches_mms.m  </postProcess_cmd>
    <x>5e-6</x>
    <replace_lines>
      <delt_init>5e-6</delt_init>
    </replace_lines>
</Test>

<Test>
    <Title>3</Title>
    <sus_cmd>mpirun -np 1 sus -mpi</sus_cmd>
    <postProcess_cmd>arches_mms.m </postProcess_cmd>
    <x>2.5e-6</x>
    <replace_lines>
      <delt_init>2.5e-6</delt_init>
    </replace_lines>
</Test>

<Test>
    <Title>4</Title>
    <sus_cmd>mpirun -np 1 sus -mpi</sus_cmd>
    <postProcess_cmd>arches_mms.m </postProcess_cmd>
    <x>1.25e-6</x>
    <replace_lines>
      <delt_init>1.25e-6</delt_init>
    </replace_lines>
</Test>

<Test>
    <Title>5</Title>
    <sus_cmd>mpirun -np 1 sus -mpi</sus_cmd>
    <postProcess_cmd>arches_mms.m </postProcess_cmd>
    <x>6.25e-7</x>
    <replace_lines>
      <delt_init>6.25e-7</delt_init>
    </replace_lines>
</Test>

</start>
