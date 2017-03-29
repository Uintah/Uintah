<?xml version="1.0" encoding="ISO-8859-1"?>
<!--______________________________________________________________________ -->
<!--  This parametric study compares the boundary fluxes against a pre computed 'gold standard' -->
<!--  First run the input file with large number of boundary flux rays.                          -->
<!--  Then copy that uda to your home directory and point the postProcessTools/RMCRT_boundFlux  -->
<!--  script to it.                                                                             -->
<!--______________________________________________________________________ -->
<start>

<upsFile>RMCRT_isoScat.ups</upsFile>

<gnuplot>
  <script>plotBoundFlux.gp</script>s
  <title>GPU::RMCRT Boundary Flux Comparison Against CPU Implementation \\n 1 timestep (41^3)</title>
  <ylabel>L2 Error</ylabel>
  <xlabel>Boundary Flux Rays</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
    <max_Timesteps>1</max_Timesteps>
    <resolution> [41,41,41]  </resolution>
  </replace_lines>
</AllTests>

<Test>
    <Title>2</Title>
    <sus_cmd> sus -gpu -nthreads 2 </sus_cmd>
    <postProcess_cmd>RMCRT_boundFlux</postProcess_cmd>
    <x>2</x>
    <replace_lines>
      <nFluxRays>          2        </nFluxRays>
    </replace_lines>
</Test>

<Test>
    <Title>4</Title>
    <sus_cmd> sus -gpu -nthreads 2 </sus_cmd>
    <postProcess_cmd>RMCRT_boundFlux</postProcess_cmd>
    <x>4</x>
    <replace_lines>
      <nFluxRays>          4        </nFluxRays>
    </replace_lines>
</Test>

<Test>
    <Title>8</Title>
    <sus_cmd> sus -gpu -nthreads 2 </sus_cmd>
    <postProcess_cmd>RMCRT_boundFlux</postProcess_cmd>
    <x>8</x>
    <replace_lines>
      <nFluxRays>          8        </nFluxRays>
    </replace_lines>
</Test>

<Test>
    <Title>16</Title>
    <sus_cmd>sus -gpu -nthreads 2 </sus_cmd>
    <postProcess_cmd>RMCRT_boundFlux</postProcess_cmd>
    <x>16</x>
    <replace_lines>
      <nFluxRays>          16        </nFluxRays>
    </replace_lines>
</Test>

<Test>
    <Title>32</Title>
    <sus_cmd>sus -gpu -nthreads 2  </sus_cmd>
    <postProcess_cmd>RMCRT_boundFlux</postProcess_cmd>
    <x>32</x>
    <replace_lines>
      <nFluxRays>          32        </nFluxRays>
    </replace_lines>
</Test>

<Test>
    <Title>64</Title>
    <sus_cmd> sus -gpu -nthreads 2 </sus_cmd>
    <postProcess_cmd>RMCRT_boundFlux</postProcess_cmd>
    <x>64</x>
    <replace_lines>
      <nFluxRays>          64        </nFluxRays>
    </replace_lines>
</Test>

<Test>
    <Title>128</Title>
    <sus_cmd> sus -gpu -nthreads 2  </sus_cmd>
    <postProcess_cmd>RMCRT_boundFlux</postProcess_cmd>
    <x>128</x>
    <replace_lines>
      <nFluxRays>          128        </nFluxRays>
    </replace_lines>
</Test>

<Test>
    <Title>256</Title>
    <sus_cmd> sus -gpu -nthreads 2  </sus_cmd>
    <postProcess_cmd> RMCRT_boundFlux</postProcess_cmd>
    <x>256</x>
    <replace_lines>
      <nFluxRays>          256        </nFluxRays>
    </replace_lines>
</Test>

<Test>
    <Title>512</Title>
    <sus_cmd> sus -gpu -nthreads 2  </sus_cmd>
    <postProcess_cmd>RMCRT_boundFlux</postProcess_cmd>
    <x>512</x>
    <replace_lines>
      <nFluxRays>          512        </nFluxRays>
    </replace_lines>
</Test>

<Test>
    <Title>1024</Title>
    <sus_cmd> sus -gpu -nthreads 2  </sus_cmd>
    <postProcess_cmd>RMCRT_boundFlux </postProcess_cmd>
    <x>1024</x>
    <replace_lines>
      <nFluxRays>          1024        </nFluxRays>
    </replace_lines>
</Test>

</start>
