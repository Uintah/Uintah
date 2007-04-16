<start>

<Test>

<Meta>
<Title>res_100</Title>
<upsFile>mms_1.ups</upsFile>
<pbsFile>mms_longrun_mod.pbs</pbsFile>
<Study>Resolution Study</Study>
<compCommand>compare_mms -ice -mms sine -v press_CC -L</compCommand>
<x>100</x>
</Meta>

<content>
       <Level>
           <Box label="1">
              <lower>        [0,0,-0.05]    </lower>
              <upper>        [6.28318530717959,6.28318530717959, 0.05]    </upper>
              <extraCells>   [0,0,1]              </extraCells>
              <patches>      [2,1,1]              </patches>
              <resolution>   [100,100,1]          </resolution>
           </Box>
           <periodic>       [1,1,0]           </periodic>
       </Level>
</content>

</Test>

<Test>

<Meta>
<Title>res_200</Title>
<upsFile>mms_1.ups</upsFile>
<pbsFile>mms_longrun_mod.pbs</pbsFile>
<Study>Resolution Study</Study>
<compCommand>compare_mms -ice -mms sine -v press_CC -L</compCommand>
<x>200</x>
</Meta>

<content>
       <Level>
           <Box label="1">
              <lower>        [0,0,-0.05]    </lower>
              <upper>        [6.28318530717959,6.28318530717959, 0.05]    </upper>
              <extraCells>   [0,0,1]              </extraCells>
              <patches>      [2,1,1]              </patches>
              <resolution>   [200,200,1]          </resolution>
           </Box>
           <periodic>       [1,1,0]           </periodic>
       </Level>
</content>

</Test>

</start>

