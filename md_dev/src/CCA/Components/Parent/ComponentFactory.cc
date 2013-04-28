/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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
#include <CCA/Components/Examples/Benchmark.h>
#include <CCA/Components/Examples/Burger.h>
#include <CCA/Components/Examples/RMCRT_Test.h>
#include <CCA/Components/Examples/ParticleTest1.h>
#include <CCA/Components/Examples/Poisson1.h>
#include <CCA/Components/Examples/Poisson2.h>
#include <CCA/Components/Examples/Poisson3.h>
#include <CCA/Components/Examples/Poisson4.h>
#include <CCA/Components/Examples/RegridderTest.h>
#include <CCA/Components/Examples/SolverTest1.h>
#include <CCA/Components/Examples/Wave.h>
#include <CCA/Components/Examples/LJPotentialTest.h>
#include <CCA/Components/ICE/AMRICE.h>
#include <CCA/Components/ICE/ICE.h>
#include <CCA/Components/ICE/impAMRICE.h>
#include <CCA/Components/MPM/AMRMPM.h>
#include <CCA/Components/MPM/ImpMPM.h>
#include <CCA/Components/MPM/RigidMPM.h>
#include <CCA/Components/MPM/SerialMPM.h>
#include <CCA/Components/MPM/ShellMPM.h>
#include <CCA/Components/MPMArches/MPMArches.h>
#include <CCA/Components/MPMICE/MPMICE.h>
#include <CCA/Components/MD/MD.h>
#include <CCA/Components/Parent/ComponentFactory.h>
#include <CCA/Components/Parent/Switcher.h>
#include <CCA/Components/PatchCombiner/PatchCombiner.h>
#include <CCA/Components/PatchCombiner/UdaReducer.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <sci_defs/uintah_defs.h>
#include <sci_defs/cuda_defs.h>

#ifdef HAVE_CUDA
#include <CCA/Components/Examples/UnifiedSchedulerTest.h>
#include <CCA/Components/Examples/GPUSchedulerTest.h>
#include <CCA/Components/Examples/PoissonGPU1.h>
#endif

#ifndef NO_WASATCH
#include <CCA/Components/Wasatch/Wasatch.h>
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
    if( !Uintah::Parallel::usingMPI() ) {
      throw ProblemSetupException("You must be using MPI when running an arches problem!", __FILE__, __LINE__);
    }
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

#ifndef NO_MD
  if (sim_comp == "md" || sim_comp == "MD") {
    return scinew MD(world);
  }
#else
  turned_off_options += "MD ";
#endif

  if (sim_comp == "burger" || sim_comp == "BURGER") {
    return scinew Burger(world);
  } 
  if (sim_comp == "wave" || sim_comp == "WAVE") {
    if (doAMR)
      return scinew AMRWave(world);
    else
      return scinew Wave(world);
  }

#ifndef NO_WASATCH
  if (sim_comp == "wasatch") {
    return scinew Wasatch::Wasatch(world);
  } 
#endif

  if (sim_comp == "poisson1" || sim_comp == "POISSON1") {
    return scinew Poisson1(world);
  }

#ifdef HAVE_CUDA
  if (sim_comp == "poissongpu1" || sim_comp == "POISSONGPU1") {
    return scinew PoissonGPU1(world);
  }
  if (sim_comp == "gpuschedulertest" || sim_comp == "GPUSCHEDULERTEST") {
    return scinew GPUSchedulerTest(world);
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
  if (sim_comp == "ljpotentialtest" || sim_comp == "LJPOTENTIALTEST") {
    return scinew LJPotentialTest(world);
  }
  if (sim_comp == "solvertest" || sim_comp == "SOLVERTEST") {
    return scinew SolverTest1(world);
  } 
  if (sim_comp == "switcher" || sim_comp == "SWITCHER") {
    return scinew Switcher(world, ps, doAMR, uda);
  } 
  if (sim_comp == "combine_patches") {
    return scinew PatchCombiner(world, uda);
  } 
  if (sim_comp == "reduce_uda") {
    return scinew UdaReducer(world, uda);
  }
  // TODO this needs to be updated
  throw ProblemSetupException("Unknown simulationComponent ('" + sim_comp + "'). Must specify -arches, -ice, -mpm, "
                              "-impm, -mpmice, -mpmarches, -burger, -wave, -poisson1, -poisson2, -poisson3 or -benchmark.\n"
                              "Note: the following components were turned off at configure time: " + turned_off_options + "\n"
                              "Make sure that the requested component is supported in this build.", __FILE__, __LINE__);
}
