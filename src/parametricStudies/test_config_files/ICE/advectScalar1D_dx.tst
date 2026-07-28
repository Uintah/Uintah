<?xml version="1.0" encoding="ISO-8859-1"?>
<start>
<upsFile>advectScalar_1D.ups</upsFile>
<gnuplot>
  <script>plotScript.gp</script>s
  <title>ICE:Advection Test X dir</title>
  <ylabel>Error</ylabel>
  <xlabel>Resolution</xlabel>
</gnuplot>

<batchScheduler>
  <template> batch.slrm   </template>
  <submissionCmd> sbatch    </submissionCmd>
  <batchReplace tag="[acct]"       value = "myAcct" />
  <batchReplace tag="[partition]"  value = "myPart" />
</batchScheduler>


<Test>
    <Title>100</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>./tools/compare_mms/compare_scalar -v</postProcess_cmd>
    <batchReplace tag="[runTime]"  value = "00:01:00" />
    <batchReplace tag="[mpiRanks]" value = "1"/>
    <x>100</x>
    <replace_lines>
       <delt_init>   2.0e-5             </delt_init>
      <resolution>   [100,1,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>200</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>./tools/compare_mms/compare_scalar -v</postProcess_cmd>
    <batchReplace tag="[runTime]"  value = "00:02:00" />
    <batchReplace tag="[mpiRanks]" value = "2"/>
    <x>200</x>
    <replace_lines>
      <delt_init>    1.0e-5             </delt_init>
      <resolution>   [200,1,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>400</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>./tools/compare_mms/compare_scalar -v</postProcess_cmd>
    <batchReplace tag="[runTime]"  value = "00:03:00" />
    <batchReplace tag="[mpiRanks]" value = "3"/>
    <x>400</x>
    <replace_lines>
      <delt_init>    5.0e-6             </delt_init>
      <resolution>   [400,1,1]          </resolution>
    </replace_lines>
</Test>
<Test>
    <Title>800</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>./tools/compare_mms/compare_scalar -v</postProcess_cmd>
    <batchReplace tag="[runTime]"  value = "00:04:00" />
    <batchReplace tag="[mpiRanks]" value = "4"/>
    <x>800</x>
    <replace_lines>
      <delt_init>    2.5e-6             </delt_init>
      <resolution>   [800,1,1]          </resolution>
    </replace_lines>
</Test>

<Test>
    <Title>1600</Title>
    <sus_cmd>sus </sus_cmd>
    <postProcess_cmd>./tools/compare_mms/compare_scalar -v</postProcess_cmd>
    <batchReplace tag="[runTime]"  value = "00:05:00" />
    <batchReplace tag="[mpiRanks]" value = "5"/>
    <x>1600</x>
    <replace_lines>
      <delt_init>    1.25e-6             </delt_init>
      <resolution>   [1600,1,1]          </resolution>
    </replace_lines>
</Test>

</start>
