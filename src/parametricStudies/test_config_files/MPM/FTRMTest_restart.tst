<?xml version="1.0" encoding="ISO-8859-1"?>


<!--______________________________________________________________________-->
<!--  This tst file shows how to run a parametric study on a restart.     -->
<!--  The user defines the restart uda, the checkpoint timestep           -->
<!--  and any changes to the input.xml or timestep.xml for all tests.     -->
<!--  Each test is defined as normal.                                     -->
<!--  This tst assumes that you've alread run                             -->
<!--          mpirun -np 2 sus inputs/MPM/FTRMTest.ups                    -->
<!--______________________________________________________________________-->

<start>

<exitOnCrash> false </exitOnCrash>


<restart_uda>
  <uda>   /uufs/chpc.utah.edu/common/home/harman-group3/harman/Builds/Fresh/ash/opt-gcc_8.5-openmpi_4.1/StandAlone/FTRMTest.uda.000     </uda>
  <checkpoint_timestep>t01146 </checkpoint_timestep>
</restart_uda>


<!--__________________________________-->
<!-- Changes made to all tests defined below -->

<AllTests>
  <replace_lines>
    <maxTime>             1.3    </maxTime>
  </replace_lines>
</AllTests>


<!--______________________________________________________________________-->
<!--                                                                      -->
<!--To see the xml paths in the file execute:   xmlstarlet el -v FTRMTest.uda.000/checkpoints/t01146/timestep.xml" -->
<Test>
    <Title>loadCurve_1</Title>
    <sus_cmd> mpirun -np 2 sus   </sus_cmd>
  
    <replace_values>
      <entry path = "Uintah_timestep/PhysicalBC/MPM/pressure/load_curve/time_point[5]/load"  value ='-1000'/>
    </replace_values>
</Test>

<Test>
    <Title>loadCurve_2</Title>
    <sus_cmd> mpirun -np 2 sus   </sus_cmd>
  
    <replace_values>
      <entry path = "Uintah_timestep/PhysicalBC/MPM/pressure/load_curve/time_point[5]/time"  value ='1.2'/>
      <entry path = "Uintah_timestep/PhysicalBC/MPM/pressure/load_curve/time_point[5]/load"  value ='-1000'/>
    </replace_values>
</Test>

</start>
