#!/usr/bin/env python3

from os import symlink,environ, system
from sys import argv,exit,platform
from helpers.runSusTests_git import runSusTests, ignorePerformanceTests, getInputsDir
from helpers.modUPS import modUPS

the_dir = "%s/%s" % ( getInputsDir(),"ARCHES" )

BrownSoot_spectral_orthog_ups  = modUPS( the_dir, "Coal/BrownSoot_spectral.ups" , ["<addOrthogonalDirs> true </addOrthogonalDirs>"])



#______________________________________________________________________
#  Test syntax: ( "folder name", "input file", # processors, "OS", ["flags1","flag2"])
#
#  OS:  Linux, Darwin, or ALL
#
#  flags:
#       gpu:                        - run test if machine is gpu enabled
#       no_uda_comparison:          - skip the uda comparisons
#       no_memoryTest:              - skip all memory checks
#       no_restart:                 - skip the restart tests
#       no_dbg:                     - skip all debug compilation tests
#       no_opt:                     - skip all optimized compilation tests
#       no_cuda:                    - skip test if this is a cuda enable build
#       do_performance_test:        - Run the performance test, log and plot simulation runtime.
#                                     (You cannot perform uda comparsions with this flag set)
#       doesTestRun:                - Checks if a test successfully runs
#       abs_tolerance=[double]      - absolute tolerance used in comparisons
#       rel_tolerance=[double]      - relative tolerance used in comparisons
#       exactComparison             - set absolute/relative tolerance = 0  for uda comparisons
#       postProcessRun              - start test from an existing uda in the checkpoints directory.  Compute new quantities and save them in a new uda
#       startFromCheckpoint         - start test from checkpoint. (/home/rt/CheckPoints/..../testname.uda.000)
#       sus_options="string"        - Additional command line options for sus command
#       compareUda_options="string" - Additional command line options for compare_uda
#
#  Notes:
#  1) The "folder name" must be the same as input file without the extension.
#  2) Performance_tests are not run on a debug build.
#______________________________________________________________________
#  Test: purpose
#  massource_con_den: a constant density cold channel flow with a constant mass source. used to test mass conservation for pressure solver (with mass source terms)
#  massource_var_den: a two-fluid mixing cold channel flow with a constant mass source, used to test mass conservation for pressure solver (with mass source terms)
#  massource_coal_DQMOM: a coal reacting channel flow with a constant mass source term (or with coal mass source term), used to test mass conservation for pressure solver
#  methane_RCCE: 3m fire using the westbrook dryer/RCCE model

# NOTE on adding tests:
    # 1. Add tests to the group to which it belongs.
    # 2. Make sure that group is included in NIGHTLYTESTS if you want it tested by the builtbot, nightlyRT, and localRT
    # 3. If the test doesn't belong in any subgroup, you must manually add it to the nightlyRT and/or localRT

PRODUCTION_TESTS_NO_COAL = [
   ("constantMMS"                       , "mms/constantMMS.ups"                                     , 1   , "All"   , ["exactComparison"])   ,
   ("almgrenMMS"                        , "mms/almgrenMMS.ups"                                      , 1   , "All"   , ["exactComparison"])   ,
   ("oned_pulse_conv"                   , "mms/oned_pulse_conv.ups"                                 , 1   , "All"   , ["exactComparison"])   ,
   ("isotropic-turbulence-decay"        , "periodicTurb/isotropic-turbulence-decay.ups"             , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("isotropic_p4_dynsmag_32"           , "periodicTurb/isotropic_p4_dynsmag_32.ups"                , 8   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("isotropic_p4_wale_32"              , "periodicTurb/isotropic_p4_wale_32.ups"                   , 8   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("isotropic_p4_sigma_32"             , "periodicTurb/isotropic_p4_sigma_32.ups"                  , 8   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("helium_1m"                         , "helium_1m.ups"                                           , 1   , "All"   , ["exactComparison"])   ,
   ("methane_fire"                      , "methane_fire.ups"                                        , 1   , "All"   , ["exactComparison"])   ,
   ("methane_fire_dRad"                 , "methane_fire_dRad.ups"                                   , 4   , "All"   , ["exactComparison"]) ,
   ("methane_fire_8patch"               , "methane_fire_8patch.ups"                                 , 8   , "All"   , ["exactComparison"])   ,
#   ("methane_fire_8patch_petscrad"      , "methane_fire_8patch_petscrad.ups"                        , 8   , "All"   , ["exactComparison"     , "no_cuda" ])   ,    #11/1/6 gpu machine doesn't have petsc
   ("dqmom_test_1"                      , "DQMOM_regression/dqmom_test_1.ups"                       , 1   , "All"   , ["exactComparison"])   ,
   ("dqmom_test_2"                      , "DQMOM_regression/dqmom_test_2.ups"                       , 1   , "All"   , ["exactComparison"])   ,
   ("dqmom_test_3"                      , "DQMOM_regression/dqmom_test_3.ups"                       , 1   , "All"   , ["exactComparison"])   ,
   ("dqmom_test_4"                      , "DQMOM_regression/dqmom_test_4.ups"                       , 1   , "All"   , ["exactComparison"])   ,
   ("dqmom_test_5"                      , "DQMOM_regression/dqmom_test_5.ups"                       , 1   , "All"   , ["exactComparison"])   ,
   ("methane_jet"                       , "ClassicMixingTables/ups/methane_jet.ups"                 , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("acidbase_jet"                      , "ClassicMixingTables/ups/acidbase_jet.ups"                , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("acidbase_jet_2D"                   , "ClassicMixingTables/ups/acidbase_jet_2D.ups"             , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("1Dtabletest"                       , "ClassicMixingTables/ups/1DTableTest.ups"                 , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("xplus_scalar_test"                 , "ScalarTests/xplus_scalar_test.ups"                       , 6   , "All"   , ["exactComparison"]) ,
   ("yplus_scalar_test"                 , "ScalarTests/yplus_scalar_test.ups"                       , 6   , "All"   , ["exactComparison"]) ,
   ("zplus_scalar_test"                 , "ScalarTests/zplus_scalar_test.ups"                       , 6   , "All"   , ["exactComparison"]) ,
   ("xminus_scalar_test"                , "ScalarTests/xminus_scalar_test.ups"                      , 6   , "All"   , ["exactComparison"]) ,
   ("yminus_scalar_test"                , "ScalarTests/yminus_scalar_test.ups"                      , 6   , "All"   , ["exactComparison"]) ,
   ("zminus_scalar_test"                , "ScalarTests/zminus_scalar_test.ups"                      , 6   , "All"   , ["exactComparison"]) ,
   ("turbulent_inlet_test"              , "DigitalFilter/TurbulentInletChannel.ups"                 , 6   , "All"   , ["exactComparison"]) ,
   ("masssource_con_den"                , "verify_masssource/source_channel_conden.ups"             , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("masssource_var_den"                , "verify_masssource/source_channel_varden.ups"             , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("heptane_pipe"                      , "heptane_pipe.ups"                                        , 1   , "All"   , ["exactComparison"])   ,
   ("coal_table_pipe"                   , "coal_table_pipe.ups"                                     , 1   , "All"   , ["exactComparison"])   ,
   ("scalar_var_1eqn"                   , "scalar_variance_1eqn.ups"                                , 4   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("scalar_var_2eqn"                   , "scalar_variance_2eqn.ups"                                , 4   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("methane_RCCE"                      , "methane_RCCE.ups"                                        , 1   , "All"   , ["exactComparison"])   ,
   ("channel_WD_CO"                     , "channel_WD_CO.ups"                                       , 1   , "All"   , ["exactComparison"])   ,
   ("DOM16"                             , "DOM16.ups"                                               , 3   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("CQMOM_1x1"                         , "CQMOM_regression/CQMOM_1x1.ups"                          , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("CQMOM_scalar_transport"            , "CQMOM_regression/CQMOM_Transport.ups"                    , 6   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("CQMOM_scalar_transport2x2x2"       , "CQMOM_regression/CQMOM_Transport_2x2x2.ups"              , 6   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("SimpleBoxPTC"                      , "CQMOM_regression/SimpleBoxPTC.ups"                       , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("PTC_2D"                            , "CQMOM_regression/PTC_2D.ups"                             , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("PTC_3D"                            , "CQMOM_regression/PTC_3D.ups"                             , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("CQMOM_4D"                          , "CQMOM_regression/CQMOM_4D.ups"                           , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("CQMOM_7D"                          , "CQMOM_regression/CQMOM_7D.ups"                           , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   #("singleJet_poly"                    , "CQMOM_regression/singleJet_poly.ups"                     , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("angledWall"                        , "CQMOM_regression/angledWall.ups"                         , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("angledWall3D"                      , "CQMOM_regression/angledWall3D.ups"                       , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("Constant_Deposition"               , "CQMOM_regression/Constant_Deposition.ups"                , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   #("CQMOM_coal_test"                   , "CQMOM_regression/CQMOM_coal_test.ups"                    , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("channel_LagPart_inlet"             , "LagrangianParticles/channel_flow_x_lagrangian_inlet.ups" , 1   , "All"   , ["exactComparison"     , "no_restart", "no_cuda"]) ,  # 11/1/16 bug with gpu support with particles
   ("task_math"                         , "task_math.ups"                                           , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("intrusion_test"                    , "intrusion_test.ups"                                      , 1   , "All"   , ["exactComparison"]) ,
   ("multi-patch-intrusion-test"        , "multi-patch-intrusion-test.ups"                          , 8   , "All"   , ["exactComparison"]) ,
   ("cloudBM24LS"                       , "cloudBM24LS.ups"                                         , 8   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("cloudBM48LS"                       , "cloudBM48LS.ups"                                         , 8   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("cloudBM80GLC"                      , "cloudBM80GLC.ups"                                        , 8   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("cloudBM80LS"                       , "cloudBM80LS.ups"                                         , 8   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("heated_plate"                      , "heated_plate.ups"                                        , 1   , "All"   , ["exactComparison"]) ,
]

PRODUCTION_COAL_TESTS = [
   ("birth_test"                        , "DQMOM_regression/birth_test.ups"                         , 1   , "All"   , ["exactComparison"])   ,
   ("upwind_birth_test"                 , "DQMOM_regression/upwind_birth_test.ups"                  , 1   , "All"   , ["exactComparison"])   ,
   ("coal_channel_FOWY"                 , "Coal/coal_channel_FOWY.ups"                              , 1   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("pcoal_drag"                        , "Coal/pcoal_drag.ups"                                     , 1   , "All"   , ["exactComparison"])   ,
   ("partMassFlow"                      , "Coal/partMassFlow.ups"                                   , 8   , "All"   , ["exactComparison"   ]) ,
   ("multibox_sweeps_coal"              , "Coal/multibox_sweeps_coal.ups"                           , 46  , "All"   , ["exactComparison"]),
   #("1GW_MFR"                           , "Coal/1GW_MFR.ups"                                        , 2   , "All"   , ["exactComparison"     , "no_restart"]) ,
   ("coal_channel_hi_vel"               , "Coal/coal_channel_hi_vel.ups"                            , 1   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("1GW_RT"                            , "Coal/1GW_RT.ups"                                         , 2   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("1GW_em_tc"                         , "Coal/1GW_em_tc.ups"                                      , 2   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("1GW_pokluda"                       , "Coal/1GW_pokluda.ups"                                    , 2   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("1GW_pokluda_np"                    , "Coal/1GW_pokluda_np.ups"                                 , 2   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("mass_energy_balance"               , "Coal/mass_energy_balance.ups"                            , 2   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("mass_energy_balance_Tfluid"        , "Coal/mass_energy_balance_Tfluid.ups"                     , 2   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("mass_energy_balance_psnox"        , "Coal/mass_energy_balance_psnox.ups"                       , 2   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("coal_RadPropsPlanck"               , "Coal/coal_RadPropsPlanck.ups"                            , 1   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("OFC4"                              , "Coal/OFC4.ups"                                           , 3   , "All"   , ["exactComparison"     ,"do_performance_test",  "no_cuda"   ]) ,
   ("OFC4c"                             , "Coal/OFC4.ups"                                           , 3   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   #("OFC4_smith"                        , "Coal/OFC4_smith.ups"                                     , 3   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("OFC_smith"                         , "Coal/OFC_smith.ups"                                      , 3   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("OFC4_hybrid"                       , "Coal/OFC4_hybrid.ups"                                    , 3   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("BrownSoot_spectral"                , "Coal/BrownSoot_spectral.ups"                             , 8   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("BrownSoot_spectral_orthog"         ,  BrownSoot_spectral_orthog_ups                            , 8   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("Coal_Nox"                          , "Coal/Coal_Nox.ups"                                       , 8   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("AtikokanU1"                        , "Coal/AtikokanU1.ups"                                     , 6   , "All"   , ["exactComparison"     , "no_cuda"]) ,
]

KOKKOS_TESTS = [
   ("kokkos-x-scalar"                   , "kokkos_solver_tests/kokkos-x-scalar.ups"                                              , 1   , "All"   , ["exactComparison"]),
   ("kokkos-y-scalar"                   , "kokkos_solver_tests/kokkos-y-scalar.ups"                                              , 1   , "All"   , ["exactComparison"]),
   ("kokkos-z-scalar"                   , "kokkos_solver_tests/kokkos-z-scalar.ups"                                              , 1   , "All"   , ["exactComparison"]),
   ("almgren-mms_conv"                  , "kokkos_solver_tests/Verification/mom/almgren-mms_conv.ups"                            , 1   , "All"   , ["exactComparison"]),
   ("almgren-mms_diff"                  , "kokkos_solver_tests/Verification/mom/almgren-mms_diff.ups"                            , 1   , "All"   , ["exactComparison"]),
   ("almgren-mmsBC"                     , "kokkos_solver_tests/Verification/mom/almgren-mmsBC.ups"                               , 1   , "All"   , ["exactComparison"]),
   ("kokkos-x-scalar_mms_diff"          , "kokkos_solver_tests/Verification/scalars/kokkos-x-scalar_mms_diff.ups"                , 1   , "All"   , ["exactComparison"]),
   ("kokkos-x-scalar_mms"               , "kokkos_solver_tests/Verification/scalars/kokkos-x-scalar_mms.ups"                     , 1   , "All"   , ["exactComparison"]),
   ("kokkos-x-scalar_mms_RK1"           , "kokkos_solver_tests/Verification/scalars/kokkos-x-scalar_mms_RK1.ups"                 , 1   , "All"   , ["exactComparison"]),
   ("kokkos-x-scalar_mms_RK2"           , "kokkos_solver_tests/Verification/scalars/kokkos-x-scalar_mms_RK2.ups"                 , 8   , "All"   , ["exactComparison"]),
   ("kokkos-x-scalar_mms_RK3"           , "kokkos_solver_tests/Verification/scalars/kokkos-x-scalar_mms_RK3.ups"                 , 8   , "All"   , ["exactComparison"]),
   ("kokkos-xy-scalar"                  , "kokkos_solver_tests/Verification/scalars/2D/kokkos-xy-scalar.ups"                     , 1   , "All"   , ["exactComparison"]),
   ("kokkos-xy-scalar-MMSBC"            , "kokkos_solver_tests/Verification/scalars/2D/kokkos-xy-scalar-MMSBC.ups"               , 1   , "All"   , ["exactComparison"]),
   ("problem3_Shunn_mms-x"              , "kokkos_solver_tests/Verification/variableDensity/problem3_Shunn_mms-x.ups"            , 4   , "All"   , ["exactComparison"]),
   ("isotropic_kokkos_wale"             , "kokkos_solver_tests/Verification/periodicTurb/isotropic_kokkos_wale.ups"              , 1   , "All"   , ["exactComparison", "no_restart"]),
#   Packing dissabled for the moment...
#   ("isotropic_kokkos_dynSmag_packed"   , "kokkos_solver_tests/Verification/periodicTurb/isotropic_kokkos_dynSmag_packed.ups"    , 8   , "All"   , ["exactComparison", "no_restart"]),
   ("isotropic_kokkos_dynSmag_unpacked" , "kokkos_solver_tests/Verification/periodicTurb/isotropic_kokkos_dynSmag_unpacked.ups"  , 8   , "All"   , ["exactComparison", "no_restart"]),
   ("char_modelps"                      , "kokkos_solver_tests/Verification/particleModels/char_modelps.ups"                     , 8   , "All"   , ["exactComparison"]),
   ("dqmom_example_char"                , "kokkos_solver_tests/Verification/particleModels/dqmom_example_char.ups"               , 8   , "All"   , ["exactComparison"]),
   ("dqmom_example"                     , "kokkos_solver_tests/dqmom_example.ups"                                                , 1   , "All"   , ["exactComparison"]),
   ("OFC_mom"                           , "kokkos_solver_tests/Verification/intrusions/OFC_mom.ups"                              , 3   , "All"   , ["exactComparison"]),
   ("helium_pressure_BC"                , "kokkos_solver_tests/Verification/variableDensity/heliumKS_pressureBC.ups"             , 1   , "All"   , ["exactComparison"]),
   ("helium_plume_rk1"                  , "kokkos_solver_tests/Verification/variableDensity/heliumKS_rk1.ups"                    , 1   , "All"   , ["exactComparison"]),
   ("helium_plume_rk2"                  , "kokkos_solver_tests/Verification/variableDensity/heliumKS_rk2.ups"                    , 1   , "All"   , ["exactComparison"]),
   ("helium_plume_rk3"                  , "kokkos_solver_tests/Verification/variableDensity/heliumKS_rk3.ups"                    , 1   , "All"   , ["exactComparison"]),
   # Handoff test was having diffs on restart. It needs to be fixed:
   ("kokkos-xy-scalar-handoff"          , "kokkos_solver_tests/Verification/scalars/2D/kokkos-xy-scalar-handoff.ups"             , 1   , "All"   , ["exactComparison", "no_restart"]),
]

RMCRT_TESTS = [
   ("rmcrt_bm1_1L"                      , "RMCRT/rmcrt_bm1_1L.ups"                , 1   , "ALL"  , [ "exactComparison" ]) ,
   ("rmcrt_bm1_1L_bounded"              , "RMCRT/rmcrt_bm1_1L_bounded.ups"        , 8   , "ALL"  , [ "exactComparison" ]) ,
   ("rmcrt_bm1_ML"                      , "RMCRT/rmcrt_bm1_ML.ups"                , 1   , "ALL"  , [ "exactComparison" ]) ,
   ("rmcrt_bm1_DO"                      , "RMCRT/rmcrt_bm1_DO.ups"                , 1   , "ALL"  , [ "exactComparison" ]) ,
   ("methane_rmcrt"                     , "RMCRT/methane_rmcrt.ups"               , 8   , "ALL"  , [ "exactComparison", "no_restart"]) ,
   ("methane_VR"                        , "RMCRT/methane_VR.ups"                  , 8   , "ALL"  , [ "exactComparison" ]) ,
   ("multibox_rmcrt_coal_1L"            , "RMCRT/multibox_rmcrt_coal_1L.ups"      , 8   , "ALL"  , [ "exactComparison" ]) ,
   ("multibox_rmcrt_coal_2L"            , "RMCRT/multibox_rmcrt_coal_2L.ups"      , 8   , "ALL"  , [ "exactComparison" ]) ,
   ("multibox_rmcrt_coal_DO"            , "RMCRT/multibox_rmcrt_coal_DO.ups"      , 8   , "ALL"  , [ "exactComparison" ]) ,

# multi-threaded RMCRT tests
   ("rmcrt_bm1_1L_thread"               , "RMCRT/rmcrt_bm1_1L.ups"                , 1   , "ALL"  , ["no_restart",  "exactComparison", "sus_options=-nthreads 4" ]) ,
   ("rmcrt_bm1_ML_thread"               , "RMCRT/rmcrt_bm1_ML.ups"                , 1   , "ALL"  , ["no_restart",  "exactComparison", "sus_options=-nthreads 4" ]) ,
   ("rmcrt_bm1_DO_thread"               , "RMCRT/rmcrt_bm1_DO.ups"                , 1   , "ALL"  , ["no_restart",  "exactComparison", "sus_options=-nthreads 8" ]) ,
   ("multibox_rmcrt_coal_1L_threaded"   , "RMCRT/multibox_rmcrt_coal_1L.ups"      , 2   , "ALL"  , [ "exactComparison", "sus_options=-nthreads 8"]) ,
   ("multibox_rmcrt_coal_2L_threaded"   , "RMCRT/multibox_rmcrt_coal_2L.ups"      , 2   , "ALL"  , [ "exactComparison", "sus_options=-nthreads 8"]) ,
   ("multibox_rmcrt_coal_DO_threaded"   , "RMCRT/multibox_rmcrt_coal_DO.ups"      , 2   , "ALL"  , [ "exactComparison", "sus_options=-nthreads 8"])
]

SWEEPS_TESTS = [
   ("methane_fire_dRad"                 , "methane_fire_dRad.ups"                 , 4   , "All"   , ["exactComparison"]) ,
   ("mass_energy_balance"               , "Coal/mass_energy_balance.ups"          , 2   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("mass_energy_balance_psnox"        , "Coal/mass_energy_balance_psnox.ups"     , 2   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("mass_energy_balance_Tfluid"        , "Coal/mass_energy_balance_Tfluid.ups"   , 2   , "All"   , ["exactComparison"     , "no_cuda"]) ,
   ("multibox_sweeps_coal"              , "Coal/multibox_sweeps_coal.ups"         , 46  , "All"   , ["exactComparison"]),
   ("BrownSoot_spectral"                , "Coal/BrownSoot_spectral.ups"           , 8   , "All"   , ["exactComparison"     , "no_cuda"]),
   ("BrownSoot_spectral_orthog"         , BrownSoot_spectral_orthog_ups           , 8   , "All"   , ["exactComparison"     , "no_cuda"])
]

NIGHTLYTESTS = [
   # The regrid test should be last.  It needs a checkpoint.  If you move it up the stack and run local_RT NIGHTLYTESTS then not all tests will run
   ("regridTestArches"                  , "regridTestArches"                                        , 8   , "Linux"  , ["startFromCheckpoint" , "no_restart"])
]

# Tests that are run during local regression testing
LOCAL_TESTS = [
]

# NO RMCRT due to the segfault on the MAC
NO_RMCRT = [
]

# An empty list for debugging purposes. Please add tests as you need them when developing, but don't actually commit anything to this list.
DEBUG = [
]

#__________________________________
# The following list is parsed by the local RT script
# and allows the user to select the tests to run
#LIST: LOCAL_TESTS KOKKOS_TESTS RMCRT_TESTS PRODUCTION_TESTS_NO_COAL PRODUCTION_COAL_TESTS SWEEPS_TESTS NIGHTLYTESTS NO_RMCRT DEBUG BUILDBOTTESTS
#__________________________________


# returns the list
def getTestList(me) :
  if me == "LOCAL_TESTS":
    TESTS = RMCRT_TESTS + PRODUCTION_COAL_TESTS + PRODUCTION_TESTS_NO_COAL + KOKKOS_TESTS
  elif me == "KOKKOS_TESTS":
    TESTS = KOKKOS_TESTS
  elif me == "RMCRT_TESTS":
    TESTS = RMCRT_TESTS
  elif me == "PRODUCTION_TESTS_NO_COAL":
    TESTS = PRODUCTION_TESTS_NO_COAL
  elif me == "PRODUCTION_COAL_TESTS":
    TESTS = PRODUCTION_COAL_TESTS
  elif me == "SWEEPS_TESTS":
    TESTS = SWEEPS_TESTS
  elif me == "NIGHTLYTESTS":
    TESTS = NIGHTLYTESTS + RMCRT_TESTS + PRODUCTION_COAL_TESTS + PRODUCTION_TESTS_NO_COAL + KOKKOS_TESTS
  elif me == "NO_RMCRT":
    TESTS = NIGHTLYTESTS + PRODUCTION_COAL_TESTS + PRODUCTION_TESTS_NO_COAL + KOKKOS_TESTS
  elif me == "DEBUG":
    TESTS = DEBUG
  elif me == "BUILDBOTTESTS":
    TESTS = ignorePerformanceTests( NIGHTLYTESTS + RMCRT_TESTS + PRODUCTION_COAL_TESTS + PRODUCTION_TESTS_NO_COAL + KOKKOS_TESTS )
  else:
    print("\nERROR:ARCHES.py  getTestList:  The test list (%s) does not exist!\n\n" % me)
    exit(1)
  return TESTS

#__________________________________

if __name__ == "__main__":

  TESTS = getTestList( environ['WHICH_TESTS'] )

  result = runSusTests(argv, TESTS, "ARCHES")

  # cleanup modified files
  command = "/bin/rm -rf %s/tmp > /dev/null 2>&1 " % (the_dir)
  system( command )

  exit( result )
