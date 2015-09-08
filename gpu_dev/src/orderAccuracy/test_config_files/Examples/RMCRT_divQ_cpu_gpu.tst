<?xml version="1.0" encoding="ISO-8859-1"?>
<!--______________________________________________________________________ -->
<!--  This parametric study compares the divQ against a pre computed 'gold standard'    -->
<!--  First run the input file with large number of divQ rays.                          -->
<!--  Then copy that uda to your home directory and point the postProcessTools/RMCRT_cpu_gpu_divQ  -->
<!--  script to it.                                                                     -->
<!--______________________________________________________________________ -->
<start>

<upsFile>RMCRT_udaInit.ups</upsFile>

<gnuplot>
  <script>plot_divQ_cpu_gpu.gp</script>
  <title>divQ Comparison \\nGPU vs CPU Implementations of RMCRT \\n</title>
  <ylabel>L2 Error</ylabel>
  <xlabel>Rays</xlabel>
</gnuplot>

<AllTests>
  <replace_lines>
    <sigmaScat>       0.0           </sigmaScat>
  </replace_lines>
</AllTests>

<Test>
    <Title>2</Title>
    <sus_cmd> sus -gpu -nthreads 2 </sus_cmd>
    <postProcess_cmd>RMCRT_divQ_cpu_gpu</postProcess_cmd>
    <x>2</x>
    <replace_lines>
      <nDivQRays>          2        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>4</Title>
    <sus_cmd> sus -gpu -nthreads 2 </sus_cmd>
    <postProcess_cmd>RMCRT_divQ_cpu_gpu</postProcess_cmd>
    <x>4</x>
    <replace_lines>
      <nDivQRays>          4        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>8</Title>
    <sus_cmd> sus -gpu -nthreads 2 </sus_cmd>
    <postProcess_cmd>RMCRT_divQ_cpu_gpu</postProcess_cmd>
    <x>8</x>
    <replace_lines>
      <nDivQRays>          8        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>16</Title>
    <sus_cmd>sus -gpu -nthreads 2 </sus_cmd>
    <postProcess_cmd>RMCRT_divQ_cpu_gpu</postProcess_cmd>
    <x>16</x>
    <replace_lines>
      <nDivQRays>          16        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>32</Title>
    <sus_cmd>sus -gpu -nthreads 2  </sus_cmd>
    <postProcess_cmd>RMCRT_divQ_cpu_gpu</postProcess_cmd>
    <x>32</x>
    <replace_lines>
      <nDivQRays>          32        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>64</Title>
    <sus_cmd> sus -gpu -nthreads 2 </sus_cmd>
    <postProcess_cmd>RMCRT_divQ_cpu_gpu</postProcess_cmd>
    <x>64</x>
    <replace_lines>
      <nDivQRays>          64        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>128</Title>
    <sus_cmd> sus -gpu -nthreads 2  </sus_cmd>
    <postProcess_cmd>RMCRT_divQ_cpu_gpu</postProcess_cmd>
    <x>128</x>
    <replace_lines>
      <nDivQRays>          128        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>256</Title>
    <sus_cmd> sus -gpu -nthreads 2  </sus_cmd>
    <postProcess_cmd> RMCRT_divQ_cpu_gpu</postProcess_cmd>
    <x>256</x>
    <replace_lines>
      <nDivQRays>          256        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>512</Title>
    <sus_cmd> sus -gpu -nthreads 2  </sus_cmd>
    <postProcess_cmd>RMCRT_divQ_cpu_gpu</postProcess_cmd>
    <x>512</x>
    <replace_lines>
      <nDivQRays>          512        </nDivQRays>
    </replace_lines>
</Test>

<Test>
    <Title>1024</Title>
    <sus_cmd> sus -gpu -nthreads 2  </sus_cmd>
    <postProcess_cmd>RMCRT_divQ_cpu_gpu </postProcess_cmd>
    <x>1024</x>
    <replace_lines>
      <nDivQRays>          1024        </nDivQRays>
    </replace_lines>
</Test>

</start>
