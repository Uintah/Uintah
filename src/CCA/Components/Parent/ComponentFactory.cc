/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/Examples/AMRWave.h>
#include <CCA/Components/Examples/AMRHeat.hpp>
#include <CCA/Components/Examples/Benchmark.h>
#include <CCA/Components/Examples/Burger.h>
#include <CCA/Components/Examples/Heat.hpp>
#include <CCA/Components/Examples/DOSweep.h>
#include <CCA/Components/Examples/RMCRT_Test.h>
#include <CCA/Components/Examples/ParticleTest1.h>
#include <CCA/Components/Examples/Poisson1.h>
#include <CCA/Components/Examples/Poisson2.h>
#include <CCA/Components/Examples/Poisson3.h>
#include <CCA/Components/Examples/Poisson4.h>
#include <CCA/Components/Examples/RegridderTest.h>
#include <CCA/Components/Examples/SolverTest1.h>
#include <CCA/Components/Examples/SolverTest2.h>
#include <CCA/Components/Examples/Wave.h>
#include <CCA/Components/ICE/AMRICE.h>
#include <CCA/Components/ICE/ICE.h>
#include <CCA/Components/ICE/impAMRICE.h>
#include <CCA/Components/MPM/AMRMPM.h>
#include <CCA/Components/MPM/ImpMPM.h>
#include <CCA/Components/MPM/RigidMPM.h>
#include <CCA/Components/MPM/SerialMPM.h>
#include <CCA/Components/MPM/ShellMPM.h>
#include <CCA/Components/MPMICE/MPMICE.h>
#include <CCA/Components/Parent/ComponentFactory.h>
#include <CCA/Components/Parent/Switcher.h>
#include <CCA/Components/ReduceUda/UdaReducer.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <sci_defs/uintah_defs.h>
#include <sci_defs/cuda_defs.h>

#ifdef HAVE_CUDA
#  include <CCA/Components/Examples/UnifiedSchedulerTest.h>
#  include <CCA/Components/Examples/PoissonGPU1.h>
#endif

#if !defined(NO_ARCHES)
#  include <CCA/Components/Arches/Arches.h>
#endif

#if !defined(NO_MPM) && !defined(NO_ARCHES)
#  include <CCA/Components/MPMArches/MPMArches.h>
#endif

#ifndef NO_WASATCH
#  include <CCA/Components/Wasatch/Wasatch.h>
#endif

#ifndef NO_FVM
#include <CCA/Components/FVM/ElectrostaticSolve.h>
#include <CCA/Components/FVM/GaussSolve.h>
#include <CCA/Components/FVM/MPNP.h>
#endif

#if !defined(NO_MPM) && !defined(NO_FVM)
#include <CCA/Components/MPMFVM/ESMPM.h>
#include <CCA/Components/MPMFVM/ESMPM2.h>
#endif

#ifndef NO_HEAT
#  include <CCA/Components/Heat/CCHeat2D.h>
#  include <CCA/Components/Heat/NCHeat2D.h>
#  include <CCA/Components/Heat/CCHeat3D.h>
#  include <CCA/Components/Heat/NCHeat3D.h>
#  include <CCA/Components/Heat/AMRCCHeat2D.h>
#  include <CCA/Components/Heat/AMRNCHeat2D.h>
#  include <CCA/Components/Heat/AMRCCHeat3D.h>
#  include <CCA/Components/Heat/AMRNCHeat3D.h>
#endif

#ifndef NO_PHASEFIELD
#  include <CCA/Components/PhaseField/PhaseField.h>
#  include <CCA/Components/PhaseField/AMRPhaseField.h>
#endif 

#include <iosfwd>
#include <string>

using namespace Uintah;
using namespace std;

UintahParallelComponent *
ComponentFactory::create( ProblemSpecP& ps, const ProcessorGroup* world, 
                          bool doAMR, string uda )
{
  string sim_comp;

  ProblemSpecP sim_ps = ps->findBlock("SimulationComponent");
  if( sim_ps ) {
    sim_ps->getAttribute( "type", sim_comp );
  }
  else {
    // This is probably a <subcomponent>, so the name of the type of
    // the component is in a different place:
    ps->getAttribute( "type", sim_comp );
  }
  if( sim_comp == "" ) {
    throw ProblemSetupException( "Could not determine the type of SimulationComponent...", __FILE__, __LINE__ );
  }

  proc0cout << "Simulation Component: \t'" << sim_comp << "'\n";

  string turned_off_options;

#ifndef NO_MPM
  if (sim_comp == "mpm" || sim_comp == "MPM") {
    return scinew SerialMPM(world);
  } 
  if (sim_comp == "rmpm" || sim_comp == "rigidmpm" || sim_comp == "RIGIDMPM") {
    return scinew RigidMPM(world);
  } 
  if (sim_comp == "amrmpm" || sim_comp == "AMRmpm" || sim_comp == "AMRMPM") {
    return scinew AMRMPM(world);
  } 
  if (sim_comp == "smpm" || sim_comp == "shellmpm" || sim_comp == "SHELLMPM") {
    return scinew ShellMPM(world);
  } 
  if (sim_comp == "impm" || sim_comp == "IMPM") {
    return scinew ImpMPM(world);
  } 
#else
  turned_off_options += "MPM ";
#endif
#ifndef NO_ICE
  if (sim_comp == "ice" || sim_comp == "ICE") {
    ProblemSpecP cfd_ps = ps->findBlock("CFD");
    ProblemSpecP ice_ps = cfd_ps->findBlock("ICE");
    ProblemSpecP imp_ps = ice_ps->findBlock("ImplicitSolver");
    bool doImplicitSolver = (imp_ps);
    
    if (doAMR){
      if(doImplicitSolver){
        return scinew impAMRICE(world);
      }else{
        return scinew AMRICE(world);
      }
    }else{
      return scinew ICE(world);
    }
  } 
#else
  turned_off_options += "ICE ";
#endif
#if !defined(NO_MPM) && !defined(NO_ICE)
  if (sim_comp == "mpmice" || sim_comp == "MPMICE") {
    return scinew MPMICE(world,STAND_MPMICE, doAMR);
  } 
  if (sim_comp == "smpmice" || sim_comp == "shellmpmice" || sim_comp == "SHELLMPMICE") {
    return scinew MPMICE(world,SHELL_MPMICE, doAMR);
  } 
  if (sim_comp == "rmpmice" || sim_comp == "rigidmpmice" || sim_comp == "RIGIDMPMICE") {
    return scinew MPMICE(world,RIGID_MPMICE, doAMR);
  } 
#else
  turned_off_options += "MPMICE ";
#endif
#ifndef NO_ARCHES
  if (sim_comp == "arches" || sim_comp == "ARCHES") {
    return scinew Arches(world,doAMR);
  } 
#else
  turned_off_options += "ARCHES ";
#endif
#if !defined(NO_MPM) && !defined(NO_ARCHES)
  if (sim_comp == "mpmarches" || sim_comp == "MPMARCHES") {
    return scinew MPMArches(world, doAMR);
  } 
#else
  turned_off_options += "MPMARCHES ";
#endif

#ifndef NO_FVM
  if (sim_comp == "electrostatic_solver"){
	  return scinew ElectrostaticSolve(world);
  }

  if(sim_comp == "gauss_solver"){
    return scinew GaussSolve(world);
  }

  if(sim_comp == "mpnp"){
    return scinew MPNP(world);
  }
#else
  turned_off_options += "FVM ";
#endif

#if !defined(NO_MPM) && !defined(NO_FVM)
  if (sim_comp == "esmpm" || sim_comp == "ESMPM") {
    return scinew ESMPM(world);
  }

  if (sim_comp == "esmpm2" || sim_comp == "ESMPM2") {
      return scinew ESMPM2(world);
    }
#else
  turned_off_options += "MPMFVM ";
#endif

  if (sim_comp == "burger" || sim_comp == "BURGER") {
    return scinew Burger(world);
  }

  if (sim_comp == "dosweep" || sim_comp == "DOSWEEP") {
    return scinew DOSweep(world);
  } 
  if (sim_comp == "wave" || sim_comp == "WAVE") {
    if (doAMR)
      return scinew AMRWave(world);
    else
      return scinew Wave(world);
  }
#ifndef NO_WASATCH
  if (sim_comp == "wasatch") {
    return scinew WasatchCore::Wasatch(world);
  } 
#endif
  if (sim_comp == "poisson1" || sim_comp == "POISSON1") {
    return scinew Poisson1(world);
  }

#ifdef HAVE_CUDA
  if (sim_comp == "poissongpu1" || sim_comp == "POISSONGPU1") {
    return scinew PoissonGPU1(world);
  }
  if (sim_comp == "unifiedschedulertest" || sim_comp == "UNIFIEDSCHEDULERTEST") {
    return scinew UnifiedSchedulerTest(world);
  }
#endif

  if (sim_comp == "regriddertest" || sim_comp == "REGRIDDERTEST") {
    return scinew RegridderTest(world);
  } 
  if (sim_comp == "poisson2" || sim_comp == "POISSON2") {
    return scinew Poisson2(world);
  } 
  if (sim_comp == "poisson3" || sim_comp == "POISSON3") {
    return scinew Poisson3(world);
  } 
  if (sim_comp == "poisson4" || sim_comp == "POISSON4") {
    return scinew Poisson4(world);
  }
  if (sim_comp == "benchmark" || sim_comp == "BENCHMARK") {
    return scinew Benchmark(world);
  } 
#ifndef NO_MODELS_RADIATION
  if (sim_comp == "RMCRT_Test") {
    return scinew RMCRT_Test(world);
  }
#else
  turned_off_options += "RMCRT_Test ";
#endif
  if (sim_comp == "particletest" || sim_comp == "PARTICLETEST") {
    return scinew ParticleTest1(world);
  } 
  if (sim_comp == "solvertest" || sim_comp == "SOLVERTEST") {
    return scinew SolverTest1(world);
  }
#ifdef HAVE_HYPRE
  if (sim_comp == "solvertest2" || sim_comp == "SOLVERTEST2") {
    return scinew SolverTest2(world);
  }
#endif
  if (sim_comp == "heat" || sim_comp == "heat") {
    if (doAMR)
      return scinew AMRHeat(world);
    else
      return scinew Heat(world);
    }
  if (sim_comp == "switcher" || sim_comp == "SWITCHER") {
    return scinew Switcher(world, ps, doAMR, uda);
  } 
  if (sim_comp == "reduce_uda") {
    return scinew UdaReducer(world, uda);
  } 
#ifndef NO_HEAT
  if ( sim_comp == "fdheat" || sim_comp == "FDHEAT" ) {
    bool doNC;
    int verbosity;
    int dimension;
    ps->findBlock ( "FDHeat" )->getWithDefault ( "node_centered", doNC, false );
    ps->findBlock ( "FDHeat" )->getWithDefault ( "verbosity", verbosity, 0 );
    ps->findBlock ( "FDHeat" )->getWithDefault ( "dimension", dimension, 2 );
    if ( doAMR ) {
      if ( doNC ) {
        if ( dimension > 2 ) {
          return scinew AMRNCHeat3D ( world, verbosity );
        } else {
          return scinew AMRNCHeat2D ( world, verbosity );
      } } else { // CC
        if ( dimension > 2 ) {
          return scinew AMRCCHeat3D ( world, verbosity );
        } else {
          return scinew AMRCCHeat2D ( world, verbosity );
      } } } else { // noAMR
      if ( doNC ) {
        if ( dimension > 2 ) {
          return scinew NCHeat3D ( world, verbosity );
        } else {
          return scinew NCHeat2D ( world, verbosity );
      } } else { // CC
        if ( dimension > 2 ) {
          return scinew CCHeat3D ( world, verbosity );
        } else {
          return scinew CCHeat2D ( world, verbosity );
  } } } }
#endif

#ifndef NO_PHASEFIELD
  if ( sim_comp == "phasefield" || sim_comp == "PHASEFIELD" ) {
    bool doNC, doTest;
    int verbosity;
    int dimension;
    ps->findBlock ( "PhaseField" )->getWithDefault ( "node_centered", doNC, false );
    ps->findBlock ( "PhaseField" )->getWithDefault ( "ws_test", doTest, false );
    ps->findBlock ( "PhaseField" )->getWithDefault ( "verbosity", verbosity, 0 );
    ps->findBlock ( "PhaseField" )->getWithDefault ( "dimension", dimension, 2 );
    if ( doAMR ) {
      if ( doTest ) {
        if ( doNC ) {
          if ( dimension > 2 ) {
            return scinew AMRNCPhaseField3DTest ( world, verbosity );
          } else {
            return scinew AMRNCPhaseField2DTest ( world, verbosity );
        } } else { // CC
          if ( dimension > 2 ) {
            return scinew AMRCCPhaseField3DTest ( world, verbosity );
          } else {
            return scinew AMRCCPhaseField2DTest ( world, verbosity );
      } } } else { // noTest
        if ( doNC ) {
          if ( dimension > 2 ) {
            return scinew AMRNCPhaseField3D ( world, verbosity );
          } else {
            return scinew AMRNCPhaseField2D ( world, verbosity );
        } } else { // CC
          if ( dimension > 2 ) {
            return scinew AMRCCPhaseField3D ( world, verbosity );
          } else {
            return scinew AMRCCPhaseField2D ( world, verbosity );
    } } } } else { // noAMR
      if ( doNC ) {
        if ( dimension > 2 ) {
          return scinew NCPhaseField3D ( world, verbosity );
       }  else {
          return scinew NCPhaseField2D ( world, verbosity );
      } } else { // CC
        if ( dimension > 2 ) {
          return scinew CCPhaseField3D ( world, verbosity );
        } else {
          return scinew CCPhaseField2D ( world, verbosity );
  } } } }
#endif
  throw ProblemSetupException("Unknown simulationComponent ('" + sim_comp + "'). Must specify -arches, -ice, -mpm, "
                              "-impm, -mpmice, -mpmarches, -burger, -wave, -poisson1, -poisson2, -poisson3 or -benchmark.\n"
                              "Note: the following components were turned off at configure time: " + turned_off_options + "\n"
                              "Make sure that the requested component is supported in this build.", __FILE__, __LINE__);
}
