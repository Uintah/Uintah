<?xml version="1.0" encoding="ISO-8859-1"?>


<!--______________________________________________________________________

To find the entry path run:

xmlstarlet el -v 1GW_pokluda_np.ups | grep xmlTag

<enamel_deposit_porosity>0.8</enamel_deposit_porosity>
Uintah_specification/CFD/ARCHES/BoundaryConditions/WallHT/model/enamel_deposit_porosity

<ash_fluid_temperature>1400.0</ash_fluid_temperature>
Uintah_specification/CFD/ARCHES/ParticleProperties/ash_fluid_temperature

______________________________________________________________________-->


<start>
<upsFile>Coal/1GW_pokluda_np.ups</upsFile>


<!--__________________________________-->
<!-- tags that will be replaced for all tests -->

<AllTests>
  <replace_lines>
    <max_Timesteps>           100         </max_Timesteps>
    <outputTimestepInterval>  10          </outputTimestepInterval>
    <resolution>              [80,40,40]  </resolution>
    <patches>                 [4,2,2]     </patches>
  </replace_lines>
  
</AllTests>


<batchScheduler>
  <template> ashBatch.slrm   </template>
  <submissionCmd> sbatch    </submissionCmd>
  <batchReplace tag="[acct]"       value = "smithp-ash-cs" />
  <batchReplace tag="[partition]"  value = "smithp-ash" />
  <batchReplace tag="[runTime]"    value = "30" />
  <batchReplace tag="[mpiRanks]"   value = "16" />
</batchScheduler>


<Test>
    <Title>A</Title>
    <sus_cmd> mpirun -np 16 sus  </sus_cmd>
    <batchReplace tag="[jobName]" value = "A" />
    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/ARCHES/BoundaryConditions/WallHT/model/enamel_deposit_porosity" value ='0.6' />
    </replace_values>
</Test>

<Test>
    <Title>B</Title>
    <sus_cmd> mpirun -np 16 sus </sus_cmd>
    <batchReplace tag="[jobName]" value = "B" />
    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/ARCHES/ParticleProperties/ash_fluid_temperature" value ='1450' />
    </replace_values>
</Test>


</start>
