<?xml version="1.0" encoding="ISO-8859-1"?>


<!--______________________________________________________________________

This parametric study varies the radiation solver tolerance.

Results from 3 runs on ash:
1e-10: Timestep 100     Time=0.0735695   Next delT=0.000746816 Wall Time=849.087   EMA=9.95539    
1e-10: Timestep 100     Time=0.0735695   Next delT=0.000746816 Wall Time=894.141   EMA=10.6325
1e-10:Timestep 100      Time=0.0735695   Next delT=0.000746816 Wall Time=829.474   EMA=10.2604

5e-9:  Timestep 100     Time=0.0708082   Next delT=0.001       Wall Time=489.564   EMA=6.28485  
5e-9:  Timestep 100     Time=0.0708082   Next delT=0.001       Wall Time=480.648   EMA=6.08239
5e-9:  Timestep 100     Time=0.0708082   Next delT=0.001       Wall Time=489.547   EMA=6.17357

1e-9:  Timestep 100     Time=0.0722388   Next delT=0.000484338 Wall Time=502.427   EMA=6.5609  
1e-9:  Timestep 100     Time=0.0722388   Next delT=0.000484338 Wall Time=496.495   EMA=6.49361
1e-9:  Timestep 100     Time=0.0722388   Next delT=0.000484338 Wall Time=498.888   EMA=6.46818   

5e-8:  Timestep 100     Time=0.0704756   Next delT=0.000545606 Wall Time=443.809   EMA=5.48713   
5e-8:  Timestep 100     Time=0.0704756   Next delT=0.000545606 Wall Time=444.073   EMA=5.50638
5e-8:  Timestep 100     Time=0.0704756   Next delT=0.000545606 Wall Time=446.067   EMA=5.5969

1e-8:  Timestep 100     Time=0.0711993   Next delT=0.000395006 Wall Time=466.613   EMA=5.97783
1e-8:  Timestep 100     Time=0.0711993   Next delT=0.000395006 Wall Time=471.954   EMA=6.04619
1e-8:  Timestep 100     Time=0.0711993   Next delT=0.000395006 Wall Time=470.039   EMA=6.00733

5e-7:  Timestep 100     Time=0.0717944   Next delT=0.000647734 Wall Time=386.66    EMA=4.61753     
5e-7:  Timestep 100     Time=0.0717944   Next delT=0.000647734 Wall Time=390.936   EMA=4.66803
5e-7:  Timestep 100     Time=0.0717944   Next delT=0.000647734 Wall Time=386.577   EMA=4.71065


1e-7:  Timestep 100     Time=0.0725696   Next delT=0.000532757 Wall Time=427.347   EMA=5.31208 
1e-7:  Timestep 100     Time=0.0725696   Next delT=0.000532757 Wall Time=428.619   EMA=5.32649
1e-7:  Timestep 100     Time=0.0725696   Next delT=0.000532757 Wall Time=433.501   EMA=5.40924

5e-6:  Timestep 100     Time=0.0695375   Next delT=0.000490124 Wall Time=328.304   EMA=3.65225 
5e-6:  Timestep 100     Time=0.0695375   Next delT=0.000490124 Wall Time=327.543   EMA=3.61661
5e-6:  Timestep 100     Time=0.0695375   Next delT=0.000490124 Wall Time=326.116   EMA=3.62103

1e-6:  Timestep 100     Time=0.0701629   Next delT=0.000670341 Wall Time=1507.84   EMA=18.0834 
1e-6:  Timestep 100     Time=0.0701629   Next delT=0.000670341 Wall Time=1501.95   EMA=17.957
1e-6:  Timestep 100     Time=0.0720095   Next delT=0.000871862 Wall Time=542.962   EMA=6.13964

To find the entry path run:

xmlstarlet el -v 1GW_pokluda_np.ups | grep <xmlTag>

______________________________________________________________________-->


<start>
<upsFile>Coal/1GW_pokluda_np.ups</upsFile>

<exitOnCrash> false </exitOnCrash>

<!--__________________________________-->
<!-- tags that will be replaced for all tests -->

<AllTests>
  <replace_lines>
    <max_Timesteps>           100          </max_Timesteps>
    <outputTimestepInterval>  0            </outputTimestepInterval>
    <resolution>              [160,80,80]  </resolution>
    <patches>                 [8,4,4]      </patches>
  </replace_lines>
  
</AllTests>


<batchScheduler>
  <template> ashBatch.slrm   </template>
  <submissionCmd> sbatch    </submissionCmd>
  <batchReplace tag="[acct]"       value = "smithp-ash-cs" />
  <batchReplace tag="[partition]"  value = "smithp-ash" />
  <batchReplace tag="[runTime]"    value = "1:00:00" />
  <batchReplace tag="[mpiRanks]"   value = "128" />
</batchScheduler>


<Test>
    <Title>1e-10</Title>
    <sus_cmd> mpirun -np 128 sus  </sus_cmd>
    <batchReplace tag="[jobName]" value = "1e-10" />
    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/ARCHES/TransportEqns/Sources/src/DORadiationModel/LinearSolver/res_tol" value ='1e-10' />
    </replace_values>
</Test>


<Test>
    <Title>5e-9</Title>
    <sus_cmd> mpirun -np 128 sus  </sus_cmd>
    <batchReplace tag="[jobName]" value = "5e-9" />
    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/ARCHES/TransportEqns/Sources/src/DORadiationModel/LinearSolver/res_tol" value ='5e-9' />
    </replace_values>
</Test>

<Test>
    <Title>1e-9</Title>
    <sus_cmd> mpirun -np 128 sus  </sus_cmd>
    <batchReplace tag="[jobName]" value = "1e-9" />
    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/ARCHES/TransportEqns/Sources/src/DORadiationModel/LinearSolver/res_tol" value ='1e-9' />
    </replace_values>
</Test>

<Test>
    <Title>5e-8</Title>
    <sus_cmd> mpirun -np 128 sus  </sus_cmd>
    <batchReplace tag="[jobName]" value = "5e-8" />
    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/ARCHES/TransportEqns/Sources/src/DORadiationModel/LinearSolver/res_tol" value ='5e-8' />
    </replace_values>
</Test>

<Test>
    <Title>1e-8</Title>
    <sus_cmd> mpirun -np 128 sus  </sus_cmd>
    <batchReplace tag="[jobName]" value = "1e-8" />
    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/ARCHES/TransportEqns/Sources/src/DORadiationModel/LinearSolver/res_tol" value ='1e-8' />
    </replace_values>
</Test>

<Test>
    <Title>5e-7</Title>
    <sus_cmd> mpirun -np 128 sus  </sus_cmd>
    <batchReplace tag="[jobName]" value = "5e-7" />
    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/ARCHES/TransportEqns/Sources/src/DORadiationModel/LinearSolver/res_tol" value ='5e-7' />
    </replace_values>
</Test>

<Test>
    <Title>1e-7</Title>
    <sus_cmd> mpirun -np 128 sus  </sus_cmd>
    <batchReplace tag="[jobName]" value = "1e-7" />
    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/ARCHES/TransportEqns/Sources/src/DORadiationModel/LinearSolver/res_tol" value ='1e-7' />
    </replace_values>
</Test>

<Test>
    <Title>5e-6</Title>
    <sus_cmd> mpirun -np 128 sus  </sus_cmd>
    <batchReplace tag="[jobName]" value = "5e-6" />
    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/ARCHES/TransportEqns/Sources/src/DORadiationModel/LinearSolver/res_tol" value ='5e-6' />
    </replace_values>
</Test>

<Test>
    <Title>1e-6</Title>
    <sus_cmd> mpirun -np 128 sus  </sus_cmd>
    <batchReplace tag="[jobName]" value = "1e-6" />
    
    <replace_values>
      <entry path = "/Uintah_specification/CFD/ARCHES/TransportEqns/Sources/src/DORadiationModel/LinearSolver/res_tol" value ='1e-6' />
    </replace_values>
</Test>

</start>
