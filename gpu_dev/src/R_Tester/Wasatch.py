#!/usr/bin/env python

from sys import argv, exit
from os import environ
from helpers.runSusTests import runSusTests, inputs_root, generatingGoldStandards
from helpers.modUPS import modUPS

the_dir = generatingGoldStandards()

if the_dir == "" :
  the_dir = "%s/Wasatch" % inputs_root()
else :
  the_dir = the_dir + "/Wasatch"

scalarequationperf_ups = modUPS( the_dir, \
                                       "ScalarTransportEquation.ups", \
                                       ["<max_Timesteps> 40 </max_Timesteps>","<resolution>[400,400,400]</resolution>","<patches>[1,1,1]</patches>"])

liddrivencavityXYRe1000adaptive_ups = modUPS( the_dir,
                                       "lid-driven-cavity-xy-Re1000.ups",
                                       ["<delt_min>0.0001</delt_min>",
                                        "<delt_max>0.1</delt_max>",
                                        "<timestep_multiplier>0.5</timestep_multiplier>",
                                       "<filebase>liddrivencavityXYRe1000adaptive.uda</filebase>"])

liddrivencavity3DRe1000rk3_ups = modUPS( the_dir, \
                                       "lid-driven-cavity-3D-Re1000.ups", \
                                       ["<TimeIntegrator> RK3SSP </TimeIntegrator>", \
                                       "<filebase>liddrivencavity3DRe1000rk3.uda</filebase>"])

lid_driven_cavity_3D_Re1000_rk2_ups = modUPS( the_dir, \
                                       "lid-driven-cavity-3D-Re1000.ups", \
                                       ["<TimeIntegrator> RK2SSP </TimeIntegrator>", \
                                       "<filebase>liddrivencavity3DRe1000rk2.uda</filebase>"])

rk2_verification_ode_ups = modUPS( the_dir, \
                                       "rk3-verification-ode.ups", \
                                       ["<TimeIntegrator> RK2SSP </TimeIntegrator>", \
                                       "<filebase>rk2-verification-ode.uda</filebase>"])

rk2_verification_timedep_source_ups = modUPS( the_dir, \
                                       "rk3-verification-timedep-source.ups", \
                                       ["<TimeIntegrator> RK2SSP </TimeIntegrator>", \
                                       "<filebase>rk2-verification-timedep-source.uda</filebase>"])

liddrivencavity3Dlaminarperf_ups = modUPS( the_dir, \
                                       "lid-driven-cavity-3D-Re1000.ups", \
                                       ["<max_Timesteps> 50 </max_Timesteps>","<resolution>[100,100,100]</resolution>","<patches>[1,1,1]</patches>"])

liddrivencavity3Dvremanperf_ups = modUPS( the_dir, \
                                       "turb-lid-driven-cavity-3D-VREMAN.ups", \
                                       ["<max_Timesteps> 50 </max_Timesteps>","<resolution>[100,100,100]</resolution>","<patches>[1,1,1]</patches>"])

liddrivencavity3Dsmagorinskyperf_ups = modUPS( the_dir, \
                                       "turb-lid-driven-cavity-3D-SMAGORINSKY.ups", \
                                       ["<max_Timesteps> 50 </max_Timesteps>","<resolution>[100,100,100]</resolution>","<patches>[1,1,1]</patches>"])

liddrivencavity3Dwaleperf_ups = modUPS( the_dir, \
                                       "turb-lid-driven-cavity-3D-WALE.ups", \
                                       ["<max_Timesteps> 50 </max_Timesteps>","<resolution>[100,100,100]</resolution>","<patches>[1,1,1]</patches>"])

scalabilitytestperf_ups = modUPS( the_dir, \
                                  "scalability-test.ups", \
                                  ["<max_Timesteps> 1000 </max_Timesteps>"])

turbulenceDir = the_dir + "/TurbulenceVerification"

decayIsotropicTurbulenceCSmag32_ups = modUPS( turbulenceDir, \
                                       "decay-isotropic-turbulence-csmag_32.ups", \
                                       ["<max_Timesteps> 10 </max_Timesteps>","<outputTimestepInterval>1</outputTimestepInterval>",'<checkpoint cycle = "2" interval = "0.001"/>'])

decayIsotropicTurbulenceCSmag64_ups = modUPS( turbulenceDir, \
                                       "decay-isotropic-turbulence-csmag_64.ups", \
                                       ["<max_Timesteps> 10 </max_Timesteps>","<outputTimestepInterval>1</outputTimestepInterval>",'<checkpoint cycle = "2" interval = "0.001"/>'])

decayIsotropicTurbulenceVreman32_ups = modUPS( turbulenceDir, \
                                       "decay-isotropic-turbulence-vreman_32.ups", \
                                       ["<max_Timesteps> 10 </max_Timesteps>","<outputTimestepInterval>1</outputTimestepInterval>",'<checkpoint cycle = "2" interval = "0.001"/>'])

decayIsotropicTurbulenceVreman64_ups = modUPS( turbulenceDir, \
                                       "decay-isotropic-turbulence-vreman_64.ups", \
                                       ["<max_Timesteps> 10 </max_Timesteps>","<outputTimestepInterval>1</outputTimestepInterval>",'<checkpoint cycle = "2" interval = "0.001"/>'])

decayIsotropicTurbulenceWale32_ups = modUPS( turbulenceDir, \
                                       "decay-isotropic-turbulence-wale_32.ups", \
                                       ["<max_Timesteps> 10 </max_Timesteps>","<outputTimestepInterval>1</outputTimestepInterval>",'<checkpoint cycle = "2" interval = "0.001"/>'])

decayIsotropicTurbulenceWale64_ups = modUPS( turbulenceDir, \
                                       "decay-isotropic-turbulence-wale_64.ups", \
                                       ["<max_Timesteps> 10 </max_Timesteps>","<outputTimestepInterval>1</outputTimestepInterval>",'<checkpoint cycle = "2" interval = "0.001"/>'])

decayIsotropicTurbulenceDSmag32_ups = modUPS( turbulenceDir, \
                                       "decay-isotropic-turbulence-dsmag_32.ups", \
                                       ["<max_Timesteps> 10 </max_Timesteps>","<outputTimestepInterval>1</outputTimestepInterval>",'<checkpoint cycle = "2" interval = "0.001"/>'])

decayIsotropicTurbulenceDSmag64_ups = modUPS( turbulenceDir, \
                                       "decay-isotropic-turbulence-dsmag_64.ups", \
                                       ["<max_Timesteps> 10 </max_Timesteps>","<outputTimestepInterval>1</outputTimestepInterval>",'<checkpoint cycle = "2" interval = "0.001"/>'])

#______________________________________________________________________
#  Test syntax: ( "folder name", "input file", # processors, "OS", ["flags1","flag2"])
#  flags:
#       gpu:                    - run test if machine is gpu enabled
#       no_uda_comparison:      - skip the uda comparisons
#       no_memoryTest:          - skip all memory checks
#       no_restart:             - skip the restart tests
#       no_dbg:                 - skip all debug compilation tests
#       no_opt:                 - skip all optimized compilation tests
#       do_performance_test:    - Run the performance test, log and plot simulation runtime.
#                                 (You cannot perform uda comparsions with this flag set)
#       doesTestRun:            - Checks if a test successfully runs
#       abs_tolerance=[double]  - absolute tolerance used in comparisons
#       rel_tolerance=[double]  - relative tolerance used in comparisons
#       exactComparison         - set absolute/relative tolerance = 0  for uda comparisons
#       startFromCheckpoint     - start test from checkpoint. (/home/csafe-tester/CheckPoints/..../testname.uda.000)
#       sus_options="string"    - Additional command line options for sus command
#
#  Notes:
#  1) The "folder name" must be the same as input file without the extension.
#  2) If the processors is > 1.0 then an mpirun command will be used
#  3) Performance_tests are not run on a debug build.
#______________________________________________________________________

DEBUGTESTS = [
  ("lid-driven-cavity-3D-Re1000",   "lid-driven-cavity-3D-Re1000.ups",   8,  "All",   ["exactComparison"] ),
  ("liddrivencavity3DRe1000rk3",   liddrivencavity3DRe1000rk3_ups,   8,  "All",  ["exactComparison","no_restart"] )  
  ]

NIGHTLYTESTS = [
#   ("scalar-transport-equation_perf",    scalarequationperf_ups,   1.1,  "Linux",  ["no_uda_comparison","no_restart","do_performance_test"] ),
#	("particle-test-driven-cavity-upper-half-IC", "particle-test-driven-cavity-upper-half-IC.ups",   4,  "Linux",   ["exactComparison","no_memoryTest"] ),
  ("varden-projection-mms-analytic",                         "varden-projection-mms-analytic.ups",              3,  "All",  ["exactComparison","no_restart","do_not_validate"] ),
	("particle-test-interpolate-to-mesh.ups", "particle-test-interpolate-to-mesh.ups",  4,  "All",   ["exactComparison","no_restart","no_memoryTest","no_dbg"] ),
	("interior-bc-test", "interior-bc-test.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-injection-multiple", "particle-test-injection-multiple.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-wall-bc-all-dir", "particle-test-wall-bc-all-dir.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-wall-bc-xdir", "particle-test-wall-bc-xdir.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-wall-bc-ydir", "particle-test-wall-bc-ydir.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-wall-bc-zdir", "particle-test-wall-bc-zdir.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),	
	("particle-test-free-fall-two-way-coupling-xdir", "particle-test-free-fall-two-way-coupling-xdir.ups",  8,  "Linux",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-free-fall-two-way-coupling-ydir", "particle-test-free-fall-two-way-coupling-ydir.ups",  8,  "Linux",   ["exactComparison","no_restart","no_memoryTest"] ),	
	("particle-test-free-fall-two-way-coupling-zdir", "particle-test-free-fall-two-way-coupling-zdir.ups",  8,  "Linux",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-geom-shape-icse", "particle-test-geom-shape-icse.ups",  1,  "Linux",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-geom-shape-flow-mickey-mouse", "particle-test-geom-shape-flow-mickey-mouse.ups",   1.1,  "Linux",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-free-fall-xdir", "particle-test-free-fall-xdir.ups",   1,  "Linux",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-free-fall-ydir", "particle-test-free-fall-ydir.ups",   1,  "Linux",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-free-fall-zdir", "particle-test-free-fall-zdir.ups",   1,  "Linux",   ["exactComparison","no_restart","no_memoryTest"] ),		
  ("turbulent-flow-over-cavity",                         "turbulent-flow-over-cavity.ups",    8,  "Linux",  ["abs_tolerance=1e-8","no_restart","no_memoryTest","no_dbg"]                                          ),
  ("coal-boiler-mini",                         "coal-boiler-mini.ups",    16,  "Linux",  ["exactComparison","no_restart","no_memoryTest"]                                          ),
  ("bc-linear-inlet-channel-flow-test",     "bc-linear-inlet-channel-flow-test.ups",             6,  "Linux",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("bc-parabolic-inlet-channel-flow-test",  "bc-parabolic-inlet-channel-flow-test.ups",             6,  "Linux",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("RMCRT-Burns-Christon", "RMCRT-Burns-Christon.ups",   8,  "Linux",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("clip-with-intrusions-test",           "clip-with-intrusions-test.ups",    4,  "Linux",  ["exactComparison","no_restart"] ),
  ("reduction-test",       "reduction-test.ups",  4,  "Linux",  ["exactComparison","no_restart"] ),
  ("lid-drive-cavity-xy-Re1000-adaptive",       liddrivencavityXYRe1000adaptive_ups,  4,  "All",  ["exactComparison","no_restart"] ),
  ("decay-isotropic-turbulence-dsmag32",       "TurbulenceVerification/"+decayIsotropicTurbulenceDSmag32_ups,  8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("decay-isotropic-turbulence-dsmag64",       "TurbulenceVerification/"+decayIsotropicTurbulenceDSmag64_ups,  8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("decay-isotropic-turbulence-csmag32",       "TurbulenceVerification/"+decayIsotropicTurbulenceCSmag32_ups,  8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("decay-isotropic-turbulence-csmag64",       "TurbulenceVerification/"+decayIsotropicTurbulenceCSmag64_ups,  8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("decay-isotropic-turbulence-vreman32",      "TurbulenceVerification/"+decayIsotropicTurbulenceVreman32_ups, 8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("decay-isotropic-turbulence-vreman64",      "TurbulenceVerification/"+decayIsotropicTurbulenceVreman64_ups, 8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("decay-isotropic-turbulence-wale32",        "TurbulenceVerification/"+decayIsotropicTurbulenceWale32_ups,   8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("decay-isotropic-turbulence-wale64",        "TurbulenceVerification/"+decayIsotropicTurbulenceWale64_ups,   8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("turbulent-inlet-test-xminus",              "turbulent-inlet-test-xminus.ups",    12,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("turb-lid-driven-cavity-3D-VREMAN_perf",    liddrivencavity3Dvremanperf_ups,   1.1,  "Linux",  ["no_uda_comparison","no_restart","do_performance_test"] ),
  ("turb-lid-driven-cavity-3D-SMAGORINSKY_perf",   liddrivencavity3Dsmagorinskyperf_ups,   1.1,  "Linux",  ["no_uda_comparison","no_restart","do_performance_test"] ),
  ("turb-lid-driven-cavity-3D-WALE_perf",      liddrivencavity3Dwaleperf_ups,   1.1,  "Linux",  ["no_uda_comparison","no_restart","do_performance_test"] ),
  ("lid-driven-cavity-3D-LAMINAR_perf",        liddrivencavity3Dlaminarperf_ups,   1.1,  "Linux",  ["no_uda_comparison","no_restart","do_performance_test"] ),
  ("intrusion_flow_past_cylinder_xy",          "intrusion_flow_past_cylinder_xy.ups",    8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("intrusion_flow_past_cylinder_xz",          "intrusion_flow_past_cylinder_xz.ups",    8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("intrusion_flow_past_cylinder_yz",          "intrusion_flow_past_cylinder_yz.ups",    8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("intrusion_flow_past_objects_xy",           "intrusion_flow_past_objects_xy.ups",    16,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("intrusion_flow_over_icse",                 "intrusion_flow_over_icse.ups",          16,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("intrusion_flow_past_oscillating_cylinder_xy",          "intrusion_flow_past_oscillating_cylinder_xy.ups",    8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("turb-lid-driven-cavity-3D-VREMAN",   "turb-lid-driven-cavity-3D-VREMAN.ups",   8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("projection_rk3_verification_dt0.01s",      "order-verification/projection_rk3_verification_dt0.01s.ups",   16,  "All",   ["exactComparison","no_restart","do_not_validate"] ),
  ("taylor-green-vortex-mms-pressure-src",      "taylor-green-vortex-mms-pressure-src.ups",   4,  "Linux",   ["exactComparison","no_restart","do_not_validate"] ),
  ("rk3-verification-ode",                     "rk3-verification-ode.ups",   1,  "Linux",   ["exactComparison","no_restart","do_not_validate"] ),
  ("rk3-verification-timedep-source",          "rk3-verification-timedep-source.ups",   1,  "Linux",   ["exactComparison","no_restart","do_not_validate"] ),
  ("bc-modifier-expression-test-multiple",   "bc-modifier-expression-test-multiple.ups",   8,  "Linux",   ["exactComparison","no_restart","do_not_validate"] ),
  ("read-from-file-test",   "read-from-file-test.ups",   8,  "Linux",   ["exactComparison","no_restart","do_not_validate"] ),
  ("channel-flow-symmetry-bc",   "channel-flow-symmetry-bc.ups",   6,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("turb-lid-driven-cavity-3D-WALE",   "turb-lid-driven-cavity-3D-WALE.ups",   8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("turb-lid-driven-cavity-3D-SMAGORINSKY",   "turb-lid-driven-cavity-3D-SMAGORINSKY.ups",   8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("turb-lid-driven-cavity-3D-scalar",   "turb-lid-driven-cavity-3D-SMAGORINSKY-scalar.ups",   8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("varden-3D-lowres-jet-IMPULSE",                  "varden-3D-lowres-jet-IMPULSE.ups",       8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("varden-projection-mms",                         "varden-projection-mms.ups",              3,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("varden-projection-2d-oscillating-periodic-mms-xy", "varden-projection-2d-oscillating-periodic-mms-xy.ups", 4,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("varden-projection-2d-oscillating-periodic-mms-xz", "varden-projection-2d-oscillating-periodic-mms-xz.ups", 4,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("varden-projection-2d-oscillating-periodic-mms-yz", "varden-projection-2d-oscillating-periodic-mms-yz.ups", 4,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("varden-projection-advection-xdir",              "varden-projection-advection-xdir.ups",   3,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("varden-projection-advection-ydir",              "varden-projection-advection-ydir.ups",   3,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("varden-projection-advection-zdir",              "varden-projection-advection-zdir.ups",   3,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("varden-projection-advection-xdir-analytic-dens","varden-projection-advection-xdir-analytic-dens.ups", 3, "Linux", ["exactComparison","no_restart","do_not_validate"] ),
  ("channel-flow-xy-xminus-pressure-outlet",   "channel-flow-xy-xminus-pressure-outlet.ups",   6,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("channel-flow-xy-xplus-pressure-outlet",    "channel-flow-xy-xplus-pressure-outlet.ups",    6,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("channel-flow-xz-zminus-pressure-outlet",   "channel-flow-xz-zminus-pressure-outlet.ups",   6,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("channel-flow-xz-zplus-pressure-outlet",    "channel-flow-xz-zplus-pressure-outlet.ups",    6,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("channel-flow-yz-yminus-pressure-outlet",   "channel-flow-yz-yminus-pressure-outlet.ups",   6,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("channel-flow-yz-yplus-pressure-outlet",    "channel-flow-yz-yplus-pressure-outlet.ups",    6,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("lid-driven-cavity-3D-Re1000",   "lid-driven-cavity-3D-Re1000.ups",   8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("liddrivencavity3DRe1000rk3",    liddrivencavity3DRe1000rk3_ups,   8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("lid-driven-cavity-xy-Re1000",   "lid-driven-cavity-xy-Re1000.ups",   4,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("lid-driven-cavity-xz-Re1000",   "lid-driven-cavity-xz-Re1000.ups",   4,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("lid-driven-cavity-yz-Re1000",   "lid-driven-cavity-yz-Re1000.ups",   4,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("hydrostatic-pressure-test",     "hydrostatic-pressure-test.ups",     8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("BasicScalarTransportEquation",  "BasicScalarTransportEquation.ups",  1,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("BasicScalarTransportEq_2L",     "BasicScalarTransportEq_2L.ups",     1,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),
  ("TabPropsInterface",             "TabPropsInterface.ups",             1,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),
  ("bc-test-mixed",                 "bc-test-mixed.ups",                 4,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("ScalarTransportEquation",       "ScalarTransportEquation.ups",       1,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("taylor-green-vortex-2d-xy",          "taylor-green-vortex-2d-xy.ups",          4,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("taylor-green-vortex-2d-xz",          "taylor-green-vortex-2d-xz.ups",          4,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("taylor-green-vortex-2d-yz",          "taylor-green-vortex-2d-yz.ups",          4,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("scalability-test",              "scalability-test.ups",              1,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("taylor-green-vortex-3d",          "taylor-green-vortex-3d.ups",          8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("bc-test-svol-xdir",             "bc-test-svol-xdir.ups",             4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),
  ("bc-test-svol-ydir",             "bc-test-svol-ydir.ups",             4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),
  ("bc-test-svol-zdir",             "bc-test-svol-zdir.ups",             4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),
  ("qmom-realizable-test",          "qmom-realizable-test.ups",          8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("qmom-test",                     "qmom-test.ups",                     4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),
  ("qmom-aggregation-test",         "qmom-aggregation-test.ups",         1,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("qmom-birth-test",               "qmom-birth-test.ups",               1,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("qmom-ostwald-test",             "qmom-ostwald-test.ups",             1,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("qmom-surface-energy-test",      "qmom-surface-energy-test.ups",      1,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("convection-test-svol-xdir",     "convection-test-svol-xdir.ups",     4,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("convection-test-svol-ydir",     "convection-test-svol-ydir.ups",     4,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("convection-test-svol-zdir",     "convection-test-svol-zdir.ups",     4,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("convection-test-svol-xdir-bc",  "convection-test-svol-xdir-bc.ups",  8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("convection-test-svol-ydir-bc",  "convection-test-svol-ydir-bc.ups",  8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("convection-test-svol-zdir-bc",  "convection-test-svol-zdir-bc.ups",  8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("convection-test-svol-mixed-bc", "convection-test-svol-mixed-bc.ups", 8,  "Linux",  ["exactComparison","no_restart","do_not_validate"] ),
  ("force-on-graph-postprocessing-test",     "force-on-graph-postprocessing-test.ups",   4,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] ),
  ("kinetic-energy-example",     "kinetic-energy-example.ups",   8,  "Linux",  ["exactComparison","no_restart"] ),
  ("rk2-verification-ode",                     rk2_verification_ode_ups,   1,  "Linux",   ["exactComparison","no_restart"] ),
  ("rk2-verification-timedep-source",          rk2_verification_timedep_source_ups,   1,  "Linux",   ["exactComparison","no_restart","do_not_validate"] ),
  ("lid-driven-cavity-3D-Re1000-rk2",   lid_driven_cavity_3D_Re1000_rk2_ups,   8,  "Linux",  ["exactComparison","no_restart"] )
]

TURBULENCETESTS =[
  ("decay-isotropic-turbulence-dsmag32",       "TurbulenceVerification/"+decayIsotropicTurbulenceDSmag32_ups,  8,  "All",  ["exactComparison"] ),
  ("decay-isotropic-turbulence-dsmag64",       "TurbulenceVerification/"+decayIsotropicTurbulenceDSmag64_ups,  8,  "All",  ["exactComparison","no_restart"] ),
  ("decay-isotropic-turbulence-csmag32",       "TurbulenceVerification/"+decayIsotropicTurbulenceCSmag32_ups,  8,  "All",  ["exactComparison"] ),
  ("decay-isotropic-turbulence-csmag64",       "TurbulenceVerification/"+decayIsotropicTurbulenceCSmag64_ups,  8,  "All",  ["exactComparison","no_restart"] ),
  ("decay-isotropic-turbulence-vreman32",      "TurbulenceVerification/"+decayIsotropicTurbulenceVreman32_ups, 8,  "All",  ["exactComparison"] ),
  ("decay-isotropic-turbulence-vreman64",      "TurbulenceVerification/"+decayIsotropicTurbulenceVreman64_ups, 8,  "All",  ["exactComparison","no_restart"] ),
  ("decay-isotropic-turbulence-wale32",        "TurbulenceVerification/"+decayIsotropicTurbulenceWale32_ups,   8,  "All",  ["exactComparison"] ),
  ("decay-isotropic-turbulence-wale64",        "TurbulenceVerification/"+decayIsotropicTurbulenceWale64_ups,   8,  "All",  ["exactComparison","no_restart"] ),
  ("turb-lid-driven-cavity-3D-WALE",           "turb-lid-driven-cavity-3D-WALE.ups",   8,  "All",  ["exactComparison","no_restart"] ),
  ("turb-lid-driven-cavity-3D-SMAGORINSKY",    "turb-lid-driven-cavity-3D-SMAGORINSKY.ups",   8,  "All",  ["exactComparison","no_restart"] ),
  ("turb-lid-driven-cavity-3D-scalar",         "turb-lid-driven-cavity-3D-SMAGORINSKY-scalar.ups",   8,  "All",  ["exactComparison","no_restart"] ),
  ("turbulent-inlet-test-xminus",              "turbulent-inlet-test-xminus.ups",    12,  "All",  ["exactComparison","no_restart"] ),
  ("turb-lid-driven-cavity-3D-VREMAN",         "turb-lid-driven-cavity-3D-VREMAN.ups",   8,  "All",  ["exactComparison","no_restart"] )
]

INTRUSIONTESTS=[
  ("coal-boiler-mini",                         "coal-boiler-mini.ups",    16,  "All",  ["exactComparison","no_restart"]               ),
  ("intrusion_flow_past_cylinder_xy",          "intrusion_flow_past_cylinder_xy.ups",    8,  "All",  ["exactComparison","no_restart"] ),
  ("intrusion_flow_past_cylinder_xz",          "intrusion_flow_past_cylinder_xz.ups",    8,  "All",  ["exactComparison","no_restart"] ),
  ("intrusion_flow_past_cylinder_yz",          "intrusion_flow_past_cylinder_yz.ups",    8,  "All",  ["exactComparison","no_restart"] ),
  ("intrusion_flow_past_objects_xy",           "intrusion_flow_past_objects_xy.ups",    16,  "All",  ["exactComparison","no_restart"] ),
  ("intrusion_flow_over_icse",                 "intrusion_flow_over_icse.ups",          16,  "All",  ["exactComparison","no_restart"] ),
  ("intrusion_flow_past_oscillating_cylinder_xy",          "intrusion_flow_past_oscillating_cylinder_xy.ups",    8,  "All",  ["exactComparison","no_restart"] ),
  ("turbulent-flow-over-cavity",                         "turbulent-flow-over-cavity.ups",    8,  "All",  ["abs_tolerance=1e-8","no_restart","no_memoryTest","no_dbg"] )
#   ("clip-with-intrusions-test",           "clip-with-intrusions-test.ups",    4,  "All",  ["exactComparison","no_restart"] )
]

PROJECTIONTESTS=[
  ("channel-flow-xy-xminus-pressure-outlet",   "channel-flow-xy-xminus-pressure-outlet.ups",   6,  "All",  ["exactComparison","no_restart"] ),
  ("channel-flow-xy-xplus-pressure-outlet",    "channel-flow-xy-xplus-pressure-outlet.ups",    6,  "All",  ["exactComparison","no_restart"] ),
  ("channel-flow-xz-zminus-pressure-outlet",   "channel-flow-xz-zminus-pressure-outlet.ups",   6,  "All",  ["exactComparison","no_restart"] ),
  ("channel-flow-xz-zplus-pressure-outlet",    "channel-flow-xz-zplus-pressure-outlet.ups",    6,  "All",  ["exactComparison","no_restart"] ),
  ("channel-flow-yz-yminus-pressure-outlet",   "channel-flow-yz-yminus-pressure-outlet.ups",   6,  "All",  ["exactComparison","no_restart"] ),
  ("channel-flow-yz-yplus-pressure-outlet",    "channel-flow-yz-yplus-pressure-outlet.ups",    6,  "All",  ["exactComparison","no_restart"] ),
  ("channel-flow-symmetry-bc",                 "channel-flow-symmetry-bc.ups",   6,  "All",   ["exactComparison","no_restart"] ),
  ("lid-driven-cavity-3D-Re1000",   "lid-driven-cavity-3D-Re1000.ups",   8,  "All",   ["exactComparison"] ),
  ("lid-driven-cavity-xy-Re1000",   "lid-driven-cavity-xy-Re1000.ups",   4,  "All",   ["exactComparison","no_restart"] ),
  ("lid-driven-cavity-xz-Re1000",   "lid-driven-cavity-xz-Re1000.ups",   4,  "All",   ["exactComparison","no_restart"] ),
  ("lid-driven-cavity-yz-Re1000",   "lid-driven-cavity-yz-Re1000.ups",   4,  "All",   ["exactComparison","no_restart"] ),
  ("hydrostatic-pressure-test",     "hydrostatic-pressure-test.ups",     8,  "All",   ["exactComparison","no_restart"] ),
  ("taylor-green-vortex-2d-xy",          "taylor-green-vortex-2d-xy.ups",          4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("taylor-green-vortex-2d-xz",          "taylor-green-vortex-2d-xz.ups",          4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("taylor-green-vortex-2d-yz",          "taylor-green-vortex-2d-yz.ups",          4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("taylor-green-vortex-mms-pressure-src",      "taylor-green-vortex-mms-pressure-src.ups",   4,  "All",   ["exactComparison","no_restart"] ),
  ("taylor-green-vortex-3d",          "taylor-green-vortex-3d.ups",          8,  "All",   ["exactComparison","no_restart","no_memoryTest"] )
]

RKTESTS=[
  ("projection_rk3_verification_dt0.01s",      "order-verification/projection_rk3_verification_dt0.01s.ups",   16,  "All",   ["exactComparison","no_restart"] ),
  ("rk3-verification-ode",                     "rk3-verification-ode.ups",   1,  "All",   ["exactComparison","no_restart"] ),
  ("rk3-verification-timedep-source",          "rk3-verification-timedep-source.ups",   1,  "All",   ["exactComparison","no_restart"] ),
  ("liddrivencavity3DRe1000rk3",   liddrivencavity3DRe1000rk3_ups,   8,  "All",  ["exactComparison","no_restart"] ),
  ("rk2-verification-ode",                     rk2_verification_ode_ups,   1,  "All",   ["exactComparison","no_restart"] ),
  ("rk2-verification-timedep-source",          rk2_verification_timedep_source_ups,   1,  "All",   ["exactComparison","no_restart","do_not_validate"] ),
  ("lid-driven-cavity-3D-Re1000-rk2",   lid_driven_cavity_3D_Re1000_rk2_ups,   8,  "All",  ["exactComparison","no_restart"] )
]

VARDENTESTS=[
  ("varden-projection-mms-analytic",                         "varden-projection-mms-analytic.ups",              3,  "All",  ["exactComparison","no_restart","do_not_validate"] ),
  ("varden-3D-lowres-jet-IMPULSE",                    "varden-3D-lowres-jet-IMPULSE.ups",   8,  "All",  ["exactComparison","no_restart"] ),
  ("varden-projection-mms",                    "varden-projection-mms.ups",   3,  "All",  ["exactComparison","no_restart"] ),
  ("varden-projection-2d-oscillating-periodic-mms-xy", "varden-projection-2d-oscillating-periodic-mms-xy.ups",   4,  "All",  ["exactComparison","no_restart"] ),
  ("varden-projection-2d-oscillating-periodic-mms-xz", "varden-projection-2d-oscillating-periodic-mms-xz.ups",   4,  "All",  ["exactComparison","no_restart"] ),
  ("varden-projection-2d-oscillating-periodic-mms-yz", "varden-projection-2d-oscillating-periodic-mms-yz.ups",   4,  "All",  ["exactComparison","no_restart"] ),
  ("varden-projection-advection-xdir",              "varden-projection-advection-xdir.ups",   3,  "All",  ["exactComparison","no_restart"] ),
  ("varden-projection-advection-ydir",              "varden-projection-advection-ydir.ups",   3,  "All",  ["exactComparison","no_restart"] ),
  ("varden-projection-advection-zdir",              "varden-projection-advection-zdir.ups",   3,  "All",  ["exactComparison","no_restart"] ),
  ("varden-projection-advection-xdir-analytic-dens","varden-projection-advection-xdir-analytic-dens.ups", 3, "All", ["exactComparison","no_restart"] )
]

MISCTESTS=[
  ("force-on-graph-postprocessing-test",     "force-on-graph-postprocessing-test.ups",   4,  "All",  ["exactComparison","no_restart","no_memoryTest"] ),
  ("kinetic-energy-example",     "kinetic-energy-example.ups",   8,  "All",  ["exactComparison","no_restart"] ) ,
  ("scalability-test",              "scalability-test.ups",              1,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("read-from-file-test",                      "read-from-file-test.ups",   8,  "All",   ["exactComparison","no_restart"] ),
  ("reduction-test",       "reduction-test.ups",  4,  "All",  ["exactComparison","no_restart"] ),
  ("lid-drive-cavity-xy-Re1000-adaptive",       liddrivencavityXYRe1000adaptive_ups,  4,  "All",  ["exactComparison","no_restart"] ),
  ("TabPropsInterface",             "TabPropsInterface.ups",             1,  "All",   ["exactComparison","no_restart","no_memoryTest"] )
]

CONVECTIONTESTS=[
  ("convection-test-svol-xdir",     "convection-test-svol-xdir.ups",     4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("convection-test-svol-ydir",     "convection-test-svol-ydir.ups",     4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("convection-test-svol-zdir",     "convection-test-svol-zdir.ups",     4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("convection-test-svol-xdir-bc",  "convection-test-svol-xdir-bc.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("convection-test-svol-ydir-bc",  "convection-test-svol-ydir-bc.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("convection-test-svol-zdir-bc",  "convection-test-svol-zdir-bc.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("convection-test-svol-mixed-bc", "convection-test-svol-mixed-bc.ups", 8,  "All",   ["exactComparison","no_restart","no_memoryTest"] )
]

BCTESTS=[
	("interior-bc-test", "interior-bc-test.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("bc-linear-inlet-channel-flow-test",     "bc-linear-inlet-channel-flow-test.ups",             6,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("bc-parabolic-inlet-channel-flow-test",  "bc-parabolic-inlet-channel-flow-test.ups",             6,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("bc-test-svol-xdir",             "bc-test-svol-xdir.ups",             4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("bc-test-svol-ydir",             "bc-test-svol-ydir.ups",             4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("bc-test-svol-zdir",             "bc-test-svol-zdir.ups",             4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("bc-modifier-expression-test-multiple",     "bc-modifier-expression-test-multiple.ups",   8,  "All",   ["exactComparison","no_restart"] ),
  ("bc-test-mixed",                 "bc-test-mixed.ups",                 4,  "All",   ["exactComparison","no_restart","no_memoryTest"] )
]

QMOMTESTS=[
  ("qmom-realizable-test",          "qmom-realizable-test.ups",          8,  "All",   ["exactComparison","no_restart"] ),
  ("qmom-test",                     "qmom-test.ups",                     4,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("qmom-aggregation-test",         "qmom-aggregation-test.ups",         1,  "All",  ["exactComparison","no_restart"] ),
  ("qmom-birth-test",               "qmom-birth-test.ups",               1,  "All",  ["exactComparison","no_restart"] ),
  ("qmom-ostwald-test",             "qmom-ostwald-test.ups",             1,  "All",  ["exactComparison","no_restart"] ),
  ("qmom-surface-energy-test",      "qmom-surface-energy-test.ups",      1,  "All",  ["exactComparison","no_restart"] )
]

SCALARTRANSPORTTESTS=[
  ("BasicScalarTransportEquation", "BasicScalarTransportEquation.ups",   1,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("BasicScalarTransportEq_2L",     "BasicScalarTransportEq_2L.ups",     1,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
  ("ScalarTransportEquation",       "ScalarTransportEquation.ups",       1,  "All",   ["exactComparison","no_restart","no_memoryTest"] )
]

GPUTESTS=[
  ("BasicScalarTransportEquation", "BasicScalarTransportEquation.ups", 1, "Linux", ["gpu", "no_restart", "no_memoryTest", "sus_options=-gpu -nthreads 2 "]),
  ("ScalarTransportEquation",      "ScalarTransportEquation.ups",      1, "Linux", ["gpu", "no_restart", "no_memoryTest", "sus_options=-gpu -nthreads 2 "])
#  ("scalability-test",             "scalability-test.ups",             1, "Linux", ["gpu", "no_restart", "no_memoryTest", "sus_options=-gpu -nthreads 2 "])
#  ("taylor-green-vortex-2d-xy",    "taylor-green-vortex-2d-xy.ups",    4, "Linux", ["gpu", "no_restart", "no_memoryTest", "sus_options=-mpi -gpu -nthreads 2 "]),
#  ("taylor-green-vortex-2d-xz",    "taylor-green-vortex-2d-xz.ups",    4, "Linux", ["gpu", "no_restart", "no_memoryTest", "sus_options=-mpi -gpu -nthreads 2 "]),
#  ("taylor-green-vortex-2d-yz",    "taylor-green-vortex-2d-yz.ups",    4, "Linux", ["gpu", "no_restart", "no_memoryTest", "sus_options=-mpi -gpu -nthreads 2 "])
]

RADIATIONTESTS=[
  ("RMCRT-Burns-Christon", "RMCRT-Burns-Christon.ups",   8,  "All",   ["exactComparison","no_restart","no_memoryTest"] )
]

PARTICLETESTS=[
	("particle-test-interpolate-to-mesh.ups", "particle-test-interpolate-to-mesh.ups",  4,  "All",   ["exactComparison","no_restart","no_memoryTest","no_dbg"] ),
	("particle-test-injection-multiple", "particle-test-injection-multiple.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-wall-bc-all-dir", "particle-test-wall-bc-all-dir.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-wall-bc-xdir", "particle-test-wall-bc-xdir.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-wall-bc-ydir", "particle-test-wall-bc-ydir.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-wall-bc-zdir", "particle-test-wall-bc-zdir.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),	
	("particle-test-free-fall-two-way-coupling-xdir", "particle-test-free-fall-two-way-coupling-xdir.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-free-fall-two-way-coupling-ydir", "particle-test-free-fall-two-way-coupling-ydir.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),	
	("particle-test-free-fall-two-way-coupling-zdir", "particle-test-free-fall-two-way-coupling-zdir.ups",  8,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-geom-shape-icse", "particle-test-geom-shape-icse.ups",  1,  "All",   ["exactComparison","no_restart","no_memoryTest","no_dbg"] ),
	("particle-test-geom-shape-flow-mickey-mouse", "particle-test-geom-shape-flow-mickey-mouse.ups",   1.1,  "All",   ["exactComparison","no_restart","no_memoryTest","no_dbg"] ),
	("particle-test-free-fall-xdir", "particle-test-free-fall-xdir.ups",   1,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-free-fall-ydir", "particle-test-free-fall-ydir.ups",   1,  "All",   ["exactComparison","no_restart","no_memoryTest"] ),
	("particle-test-free-fall-zdir", "particle-test-free-fall-zdir.ups",   1,  "All",   ["exactComparison","no_restart","no_memoryTest"] )
]

#  ("radprops",                      "RadPropsInterface.ups",             2,  "Linux",  ["exactComparison","no_restart","no_memoryTest"] )

#__________________________________
# The following list is parsed by the local RT script
# and allows the user to select the tests to run
#LIST: LOCALTESTS GPUTESTS BCTESTS CONVECTIONTESTS DEBUGTESTS INTRUSIONTESTS MISCTESTS NIGHTLYTESTS PARTICLETESTS PROJECTIONTESTS QMOMTESTS RADIATIONTESTS RKTESTS SCALARTRANSPORTTESTS TURBULENCETESTS VARDENTESTS
#__________________________________
ALLTESTS = RADIATIONTESTS + TURBULENCETESTS + INTRUSIONTESTS + PROJECTIONTESTS + RKTESTS + VARDENTESTS + MISCTESTS + CONVECTIONTESTS + BCTESTS + QMOMTESTS + SCALARTRANSPORTTESTS + PARTICLETESTS


# returns the list
def getTestList(me) :
  if me == "LOCALTESTS":
    TESTS = ALLTESTS
  elif me == "GPUTESTS":
    TESTS = GPUTESTS
  elif me == "DEBUGTESTS":
    TESTS = DEBUGTESTS
  elif me == "TURBULENCETESTS":
    TESTS = TURBULENCETESTS
  elif me == "PROJECTIONTESTS":
    TESTS = PROJECTIONTESTS
  elif me == "RKTESTS":
    TESTS = RKTESTS
  elif me == "VARDENTESTS":
    TESTS = VARDENTESTS
  elif me == "MISCTESTS":
    TESTS = MISCTESTS
  elif me == "CONVECTIONTESTS":
    TESTS = CONVECTIONTESTS
  elif me == "BCTESTS":
    TESTS = BCTESTS
  elif me == "QMOMTESTS":
    TESTS = QMOMTESTS
  elif me == "SCALARTRANSPORTTESTS":
    TESTS = SCALARTRANSPORTTESTS
  elif me == "NIGHTLYTESTS":
    TESTS = NIGHTLYTESTS
  elif me == "INTRUSIONTESTS":
    TESTS = INTRUSIONTESTS
  elif me == "RADIATIONTESTS":
    TESTS = RADIATIONTESTS
  elif me == "PARTICLETESTS":
    TESTS = PARTICLETESTS   
  else:
    print "\nERROR:Wasatch.py  getTestList:  The test list (%s) does not exist!\n\n" % me
    exit(1)
  return TESTS



# TSAAD: As an alternative to the annoying list of if-statements above, consider the following cleaner code... maybe we'll adopt
# this in the near future
# ALLTESTS = TURBULENCETESTS + INTRUSIONTESTS + PROJECTIONTESTS + RKTESTS + VARDENTESTS + MISCTESTS + CONVECTIONTESTS + BCTESTS + QMOMTESTS + SCALARTRANSPORTTESTS
# LOCALTESTS + GPUTESTS = ALLTESTS
#
# TESTNAMES=["LOCALTESTS","GPUTESTS","DEBUGTESTS","NIGHTLYTESTS","TURBULENCETESTS","INTRUSIONTESTS","PROJECTIONTESTS","RKTESTS","VARDENTESTS","MISCTESTS","CONVECTIONTESTS","BCTESTS","QMOMTESTS","SCALARTRANSPORTTESTS"]
# TESTSDICTIONARY={}
# for testname in TESTNAMES:
# 	TESTSDICTIONARY[testname]=eval(testname)
#
# # returns the list
# def getTestList(me) :
# 	return TESTSDICTIONARY[me]



#__________________________________
if __name__ == "__main__":

  TESTS = getTestList( environ['WHICH_TESTS'] )

  result = runSusTests(argv, TESTS, "Wasatch")
  exit( result )

