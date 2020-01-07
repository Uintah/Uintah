<?xml version="1.0" encoding="ISO-8859-1"?>


<!--______________________________________________________________________

This parametric study varies the CFL and diffusionKnob 

To find the entry path run:

xmlstarlet el -v DNS_Moser_Re_tau180_A.ups | grep <xmlTag>

______________________________________________________________________-->


<start>
<upsFile>DNS_Moser_Re_tau180_A.ups</upsFile>

<exitOnCrash> false </exitOnCrash>

<!--__________________________________-->
<!-- tags that will be replaced for all tests -->

<AllTests>
</AllTests>


<batchScheduler>
  <template> batch.slrm   </template>
  <submissionCmd> sbatch    </submissionCmd>
  <batchReplace tag="[acct]"       value = "efd-np" />
  <batchReplace tag="[partition]"  value = "efd-np" />
  <batchReplace tag="[runTime]"    value = "12:00:00" />
  <batchReplace tag="[mpiRanks]"   value = "32"/>
</batchScheduler>

<Test>
    <Title>0.1_1.0</Title>
    <sus_cmd> mpirun  -np 32 sus -svnDiff -svnStat </sus_cmd>    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/cfl" value ='0.1' />
      <entry path = "Uintah_specification/CFD/ICE/TimeStepControl/knob_for_diffusion" value ='1.0' />
    </replace_values>
</Test>

<Test>
    <Title>0.1_0.8</Title>
    <sus_cmd> mpirun  -np 32 sus  -svnDiff -svnStat </sus_cmd>    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/cfl" value ='0.1' />
      <entry path = "Uintah_specification/CFD/ICE/TimeStepControl/knob_for_diffusion" value ='0.8' />
    </replace_values>
</Test>

<Test>
    <Title>0.1_0.6</Title>
    <sus_cmd> mpirun  -np 32 sus  -svnDiff -svnStat </sus_cmd>    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/cfl" value ='0.1' />
      <entry path = "Uintah_specification/CFD/ICE/TimeStepControl/knob_for_diffusion" value ='0.6' />
    </replace_values>
</Test>

<Test>
    <Title>0.2_1.0</Title>
    <sus_cmd> mpirun  -np 32 sus  -svnDiff -svnStat </sus_cmd>    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/cfl" value ='0.2' />
      <entry path = "Uintah_specification/CFD/ICE/TimeStepControl/knob_for_diffusion" value ='1.0' />
    </replace_values>
</Test>

<Test>
    <Title>0.2_0.8</Title>
    <sus_cmd> mpirun  -np 32 sus  -svnDiff -svnStat </sus_cmd>    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/cfl" value ='0.2' />
      <entry path = "Uintah_specification/CFD/ICE/TimeStepControl/knob_for_diffusion" value ='0.8' />
    </replace_values>
</Test>

<Test>
    <Title>0.2_0.6</Title>
    <sus_cmd> mpirun  -np 32 sus  -svnDiff -svnStat </sus_cmd>    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/cfl" value ='0.2' />
      <entry path = "Uintah_specification/CFD/ICE/TimeStepControl/knob_for_diffusion" value ='0.6' />
    </replace_values>
</Test>


<Test>
    <Title>0.3_1.0</Title>
    <sus_cmd> mpirun  -np 32 sus  -svnDiff -svnStat </sus_cmd>    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/cfl" value ='0.3' />
      <entry path = "Uintah_specification/CFD/ICE/TimeStepControl/knob_for_diffusion" value ='1.0' />
    </replace_values>
</Test>

<Test>
    <Title>0.3_0.8</Title>
    <sus_cmd> mpirun  -np 32 sus  -svnDiff -svnStat </sus_cmd>    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/cfl" value ='0.3' />
      <entry path = "Uintah_specification/CFD/ICE/TimeStepControl/knob_for_diffusion" value ='0.8' />
    </replace_values>
</Test>

<Test>
    <Title>0.4_0.6</Title>
    <sus_cmd> mpirun  -np 32 sus  -svnDiff -svnStat </sus_cmd>    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/cfl" value ='0.4' />
      <entry path = "Uintah_specification/CFD/ICE/TimeStepControl/knob_for_diffusion" value ='0.6' />
    </replace_values>
</Test>

<Test>
    <Title>0.4_1.0</Title>
    <sus_cmd> mpirun  -np 32 sus  -svnDiff -svnStat </sus_cmd>    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/cfl" value ='0.4' />
      <entry path = "Uintah_specification/CFD/ICE/TimeStepControl/knob_for_diffusion" value ='1.0' />
    </replace_values>
</Test>

<Test>
    <Title>0.4_0.8</Title>
    <sus_cmd> mpirun  -np 32 sus  -svnDiff -svnStat </sus_cmd>    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/cfl" value ='0.4' />
      <entry path = "Uintah_specification/CFD/ICE/TimeStepControl/knob_for_diffusion" value ='0.8' />
    </replace_values>
</Test>

<Test>
    <Title>0.4_0.6</Title>
    <sus_cmd> mpirun  -np 32 sus  -svnDiff -svnStat </sus_cmd>    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/cfl" value ='0.4' />
      <entry path = "Uintah_specification/CFD/ICE/TimeStepControl/knob_for_diffusion" value ='0.6' />
    </replace_values>
</Test>

</start>
