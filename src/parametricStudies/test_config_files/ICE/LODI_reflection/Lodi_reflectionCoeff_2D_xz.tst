<?xml version="1.0" encoding="ISO-8859-1"?>
<!--
 2D (x-z plane) variant: sweeps sigma for the LODI outflow boundary
 condition's reflection coefficient |R| (reflected-to-incident acoustic
 amplitude ratio) under a genuinely 2D, radially-expanding pulse rather
 than the 1D family's planar one. See the companion .ups file for the
 full geometry/timing argument and why -index/-window are needed here
 (the probe isn't on the thin-direction centerline at index 0, and the
 reflection-arrival window differs from the 1D family's).

 References:
 [1] J.C. Sutherland, C.A. Kennedy, "Improved boundary conditions for
     viscous, reacting, compressible flows," Journal of Computational
     Physics, 191(2):502-524, 2003. (doi:10.1016/j.jcp.2003.06.001)
 [2] T.J. Poinsot, S.K. Lele, "Boundary conditions for direct simulations
     of compressible viscous flows," Journal of Computational Physics,
     101(1):104-129, 1992. (doi:10.1016/0021-9991(92)90046-2)
 [3] D.H. Rudy, J.C. Strikwerda, "A nonreflecting outflow boundary
     condition for subsonic Navier-Stokes calculations," Journal of
     Computational Physics, 36(1):55-70, 1980.
     (doi:10.1016/0021-9991(80)90174-6)
 [4] C.S. Yoo, H.G. Im, "Characteristic boundary conditions for
     simulations of compressible reacting flows with multi-dimensional,
     viscous and reaction effects," Combustion Theory and Modelling,
     11(2):259-286, 2007. (doi:10.1080/13647830600898995)
-->
<start>
<upsFile>LODI_reflection/Lodi_reflectionCoeff_2D_xz.ups</upsFile>

<gnuplot>
  <script>plotScript_reflectionCoeff.gp</script>
  <title>LODI outflow reflection coefficient vs sigma (2D, x-z plane)</title>
  <ylabel>|R|  (reflected-to-incident acoustic amplitude ratio)</ylabel>
  <xlabel>sigma</xlabel>
</gnuplot>

<Test>
    <Title>0.0</Title>
    <sus_cmd>mpirun -n 16 sus </sus_cmd>
    <postProcess_cmd>compute_LODI_reflectionCoeff.py -dir x -index 160 0 100 -window 0.0018 0.0028</postProcess_cmd>
    <x>0.0</x>
    <replace_lines>
       <sigma>  0.0  </sigma>
    </replace_lines>
</Test>

<Test>
    <Title>0.05</Title>
    <sus_cmd>mpirun -n 16 sus </sus_cmd>
    <postProcess_cmd>compute_LODI_reflectionCoeff.py -dir x -index 160 0 100 -window 0.0018 0.0028</postProcess_cmd>
    <x>0.05</x>
    <replace_lines>
       <sigma>  0.05  </sigma>
    </replace_lines>
</Test>

<Test>
    <Title>0.1</Title>
    <sus_cmd>mpirun -n 16 sus </sus_cmd>
    <postProcess_cmd>compute_LODI_reflectionCoeff.py -dir x -index 160 0 100 -window 0.0018 0.0028</postProcess_cmd>
    <x>0.1</x>
    <replace_lines>
       <sigma>  0.1    </sigma>
    </replace_lines>
</Test>

<Test>
    <Title>0.27</Title>
    <sus_cmd>mpirun -n 16 sus </sus_cmd>
    <postProcess_cmd>compute_LODI_reflectionCoeff.py -dir x -index 160 0 100 -window 0.0018 0.0028</postProcess_cmd>
    <x>0.27</x>
    <replace_lines>
       <sigma>  0.27    </sigma>
    </replace_lines>
</Test>

<Test>
    <Title>0.5</Title>
    <sus_cmd>mpirun -n 16 sus </sus_cmd>
    <postProcess_cmd>compute_LODI_reflectionCoeff.py -dir x -index 160 0 100 -window 0.0018 0.0028</postProcess_cmd>
    <x>0.5</x>
    <replace_lines>
       <sigma>  0.5     </sigma>
    </replace_lines>
</Test>

<Test>
    <Title>1.0</Title>
    <sus_cmd>mpirun -n 16 sus </sus_cmd>
    <postProcess_cmd>compute_LODI_reflectionCoeff.py -dir x -index 160 0 100 -window 0.0018 0.0028</postProcess_cmd>
    <x>1.0</x>
    <replace_lines>
       <sigma>  1.0     </sigma>
    </replace_lines>
</Test>

<Test>
    <Title>2.0</Title>
    <sus_cmd>mpirun -n 16 sus </sus_cmd>
    <postProcess_cmd>compute_LODI_reflectionCoeff.py -dir x -index 160 0 100 -window 0.0018 0.0028</postProcess_cmd>
    <x>2.0</x>
    <replace_lines>
       <sigma>  2.0     </sigma>
    </replace_lines>
</Test>

</start>
