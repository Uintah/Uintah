/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/Parent/ApplicationFactory.h>
#include <CCA/Components/Parent/Switcher.h>
#include <CCA/Components/PostProcessUda/PostProcessUda.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>

#include <sci_defs/uintah_defs.h>
#include <sci_defs/cuda_defs.h>

#ifndef NO_ARCHES
#  include <CCA/Components/Arches/Arches.h>
#endif

#ifndef NO_EXAMPLES

#if defined (HAVE_CUDA) && !defined(UINTAH_ENABLE_KOKKOS)
#  include <CCA/Components/Examples/UnifiedSchedulerTest.h>
#endif

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
#include <CCA/Components/Examples/PortableDependencyTest.h>
#include <CCA/Components/Examples/PortableDependencyTest1.h>
#include <CCA/Components/Examples/GPUResizeTest1.h>
#include <CCA/Components/Examples/RegridderTest.h>
#include <CCA/Components/Examples/SolverTest1.h>
#include <CCA/Components/Examples/SolverTest2.h>
#include <CCA/Components/Examples/Wave.h>
#endif

#ifndef NO_FVM
#include <CCA/Components/FVM/ElectrostaticSolve.h>
#include <CCA/Components/FVM/GaussSolve.h>
#include <CCA/Components/FVM/MPNP.h>
#endif

#ifndef NO_ICE
#include <CCA/Components/ICE/AMRICE.h>
#include <CCA/Components/ICE/ICE.h>
#include <CCA/Components/ICE/impAMRICE.h>
#endif

#ifndef NO_MPM
#include <CCA/Components/MPM/AMRMPM.h>
#include <CCA/Components/MPM/ImpMPM.h>
#include <CCA/Components/MPM/RigidMPM.h>
#include <CCA/Components/MPM/SerialMPM.h>
#include <CCA/Components/MPM/ShellMPM.h>
#include <CCA/Components/MPM/SingleFieldMPM.h>
#endif

#if !defined(NO_MPM) && !defined(NO_ARCHES)
#  include <CCA/Components/MPMArches/MPMArches.h>
#endif

#if !defined(NO_MPM) && !defined(NO_FVM)
#include <CCA/Components/MPMFVM/ESMPM.h>
#include <CCA/Components/MPMFVM/ESMPM2.h>
#endif

#if !defined(NO_MPM) && !defined(NO_ICE)
#include <CCA/Components/MPMICE/MPMICE.h>
#endif

#ifndef NO_PHASEFIELD
#  include <CCA/Components/PhaseField/Applications/ApplicationFactory.h>
#endif 

#ifndef NO_WASATCH
#  include <CCA/Components/Wasatch/Wasatch.h>
#endif

#include <iosfwd>
#include <string>

using namespace Uintah;

UintahParallelComponent *
ApplicationFactory::create(       ProblemSpecP     & prob_spec
                          , const ProcessorGroup   * myworld
                          , const MaterialManagerP   materialManager
                          , const std::string      & uda
                          )
{
  bool doAMR = false;

  // Check for an AMR attribute with the grid.
  ProblemSpecP grid_ps = prob_spec->findBlock("Grid");

  if (grid_ps) {
    grid_ps->getAttribute("doAMR", doAMR);
  }

  // If the AMR block is defined default to turning AMR on.
  ProblemSpecP amr_ps = prob_spec->findBlock("AMR");

  if (amr_ps) {
    doAMR = true;
  }

  std::string sim_comp;

  ProblemSpecP sim_ps = prob_spec->findBlock("SimulationComponent");
  if (sim_ps) {
    sim_ps->getAttribute("type", sim_comp);
  }
  else {
    // This is probably a <subcomponent>, so the name of the type of the component is in a different place:
    prob_spec->getAttribute("type", sim_comp);
  }

  if (sim_comp == "") {
    throw ProblemSetupException("ERROR<Application>: Could not determine the type of the SimulationComponent.", __FILE__, __LINE__);
  }

  proc0cout << "Application Component: \t'" << sim_comp << "'\n";

  std::string turned_on_options;

  //----------------------------

#ifndef NO_ARCHES
  if (sim_comp == "arches" || sim_comp == "ARCHES") {
    return scinew Arches(myworld, materialManager);
  }
  else {
    turned_on_options += "arches ";
  }
#endif

  //----------------------------

#ifndef NO_FVM
  if (sim_comp == "electrostatic_solver") {
    return scinew ElectrostaticSolve(myworld, materialManager);
  }
  else {
    turned_on_options += "electrostatic_solver ";
  }

  if(sim_comp == "gauss_solver") {
    return scinew GaussSolve(myworld, materialManager);
  }
  else {
    turned_on_options += "gauss_solver ";
  }

  if(sim_comp == "mpnp") {
    return scinew MPNP(myworld, materialManager);
  }
  else {
    turned_on_options += "mpnp ";
  }
#endif

  //----------------------------

#ifndef NO_ICE
  if (sim_comp == "ice" || sim_comp == "ICE") {
    ProblemSpecP cfd_ps = prob_spec->findBlock("CFD");
    ProblemSpecP ice_ps = cfd_ps->findBlock("ICE");
    ProblemSpecP imp_ps = ice_ps->findBlock("ImplicitSolver");
    bool doImplicitSolver = (imp_ps);

    if (doAMR) {
      if(doImplicitSolver) {
        return scinew impAMRICE(myworld, materialManager);
      }
      else {
        return scinew AMRICE(myworld, materialManager);
      }
    }
    else {
      return scinew ICE(myworld, materialManager);
    }
  }
  else {
    turned_on_options += "ice ";
  }
#endif

  //----------------------------

#ifndef NO_MPM
  if (sim_comp == "mpm" || sim_comp == "MPM") {
    return scinew SerialMPM(myworld, materialManager);
  }
  else {
    turned_on_options += "mpm ";
  }

  if (sim_comp == "rmpm" || sim_comp == "rigidmpm" || sim_comp == "RIGIDMPM") {
    return scinew RigidMPM(myworld, materialManager);
  }
  else {
    turned_on_options += "rigidmpm ";
  }

  if (sim_comp == "amrmpm" || sim_comp == "AMRmpm" || sim_comp == "AMRMPM") {
    return scinew AMRMPM(myworld, materialManager);
  }
  else {
    turned_on_options += "amrmpm ";
  }

  if (sim_comp == "sfmpm" || sim_comp == "SFmpm" || sim_comp == "SFMPM") {
    return scinew SingleFieldMPM(myworld, materialManager);
  }
  else {
    turned_on_options += "sfmpm ";
  }

  if (sim_comp == "smpm" || sim_comp == "shellmpm" || sim_comp == "SHELLMPM") {
    return scinew ShellMPM(myworld, materialManager);
  }
  else {
    turned_on_options += "shellmpm ";
  }

  if (sim_comp == "impm" || sim_comp == "IMPM") {
    return scinew ImpMPM(myworld, materialManager);
  }
  else {
    turned_on_options += "impm ";
  }
#endif

  //----------------------------

#if !defined(NO_MPM) && !defined(NO_ARCHES)
  if (sim_comp == "mpmarches" || sim_comp == "MPMARCHES") {
    return scinew MPMArches(myworld, materialManager);
  }
  else {
    turned_on_options += "mpmarches ";
  }
#endif

  //----------------------------

#if !defined(NO_MPM) && !defined(NO_FVM)
  if (sim_comp == "esmpm" || sim_comp == "ESMPM") {
    return scinew ESMPM(myworld, materialManager);
  }
  else {
    turned_on_options += "esmpms ";
  }

  if (sim_comp == "esmpm2" || sim_comp == "ESMPM2") {
    return scinew ESMPM2(myworld, materialManager);
  }
  else {
    turned_on_options += "esmpm2 ";
  }
#endif

  //----------------------------

#if !defined(NO_MPM) && !defined(NO_ICE)
  if (sim_comp == "mpmice" || sim_comp == "MPMICE") {
    return scinew MPMICE(myworld, materialManager, STAND_MPMICE, doAMR);
  } 
  else {
    turned_on_options += "mpmice ";
  }

  if (sim_comp == "smpmice" || sim_comp == "shellmpmice" || sim_comp == "SHELLMPMICE") {
    return scinew MPMICE(myworld, materialManager, SHELL_MPMICE, doAMR);
  } 
  else {
    turned_on_options += "shellmpmice ";
  }

  if (sim_comp == "rmpmice" || sim_comp == "rigidmpmice" || sim_comp == "RIGIDMPMICE") {
    return scinew MPMICE(myworld, materialManager, RIGID_MPMICE, doAMR);
  } 
  else {
    turned_on_options += "rigidmpmice ";
  }
#endif

  //----------------------------

#ifndef NO_PHASEFIELD
  if ( sim_comp == "phasefield" || sim_comp == "PHASEFIELD" ) {
    return PhaseField::ApplicationFactory::create ( myworld, materialManager, prob_spec->findBlock ( "PhaseField" ), doAMR );
  }
#endif

  //----------------------------

#ifndef NO_WASATCH
  if (sim_comp == "wasatch") {
    return scinew WasatchCore::Wasatch(myworld, materialManager);
  }
  else {
    turned_on_options += "wasatch ";
  }
#endif

  //----------------------------

  if (sim_comp == "switcher" || sim_comp == "SWITCHER") {
    return scinew Switcher(myworld, materialManager, prob_spec, uda);
  } 
  else if (!turned_on_options.empty() ) {
    turned_on_options += "switcher ";
  }

  if (sim_comp == "postProcessUda") {
    return scinew PostProcessUda(myworld, materialManager, uda);
  } 
  else {
    turned_on_options += "postProcessUda ";
  }

  //----------------------------
  
#ifndef NO_EXAMPLES
  if (sim_comp == "benchmark" || sim_comp == "BENCHMARK") {
    return scinew Benchmark(myworld, materialManager);
  }
  else {
    turned_on_options += "benchmark ";
  }
  
  if (sim_comp == "burger" || sim_comp == "BURGER") {
    return scinew Burger(myworld, materialManager);
  }
  else {
    turned_on_options += "burger ";
  }
  
  if (sim_comp == "dosweep" || sim_comp == "DOSWEEP") {
    return scinew DOSweep(myworld, materialManager);
  } 
  else {
    turned_on_options += "dosweep ";
  }
  
  if (sim_comp == "heat" || sim_comp == "heat") {
    if (doAMR) {
      return scinew AMRHeat(myworld, materialManager);
    }
    else {
      return scinew Heat(myworld, materialManager);
    }
  }
  else {
    turned_on_options += "heat ";
  }
  
  if (sim_comp == "particletest" || sim_comp == "PARTICLETEST") {
    return scinew ParticleTest1(myworld, materialManager);
  } 
  else {
    turned_on_options += "particletest ";
  }
  
  if (sim_comp == "poisson1" || sim_comp == "POISSON1") {
    return scinew Poisson1(myworld, materialManager);
  }
  else {
    turned_on_options += "poisson1 ";
  }
  
  if (sim_comp == "poisson2" || sim_comp == "POISSON2") {
    return scinew Poisson2(myworld, materialManager);
  } 
  else {
    turned_on_options += "poisson2 ";
  }

  if (sim_comp == "poisson3" || sim_comp == "POISSON3") {
    return scinew Poisson3(myworld, materialManager);
  } 
  else {
    turned_on_options += "poisson3 ";
  }

  if (sim_comp == "poisson4" || sim_comp == "POISSON4") {
    return scinew Poisson4(myworld, materialManager);
  }
  else {
    turned_on_options += "poisson4 ";
  }

  if (sim_comp == "portabledependencytest" || sim_comp == "PORTABLEDEPENDENCYTEST") {
    return scinew PortableDependencyTest(myworld, materialManager);
  }
  else {
    turned_on_options += "portabledependencytest ";
  }

  if (sim_comp == "portabledependencytest1" || sim_comp == "PORTABLEDEPENDENCYTEST1") {
    return scinew PortableDependencyTest1(myworld, materialManager);
  }
  else {
    turned_on_options += "portabledependencytest1 ";
  }

  if (sim_comp == "gpuresizetest1" || sim_comp == "GPURESIZETEST1") {
    return scinew GPUResizeTest1(myworld, materialManager);
  }
  else {
    turned_on_options += "gpuresizetest1 ";
  }

#ifndef NO_MODELS_RADIATION
  if (sim_comp == "RMCRT_Test") {
    return scinew RMCRT_Test(myworld, materialManager);
  }
  else {
    turned_on_options += "RMCRT_Test ";
  }
#endif
  
  if (sim_comp == "regriddertest" || sim_comp == "REGRIDDERTEST") {
    return scinew RegridderTest(myworld, materialManager);
  }

  if (sim_comp == "solvertest" || sim_comp == "SOLVERTEST") {
    return scinew SolverTest1(myworld, materialManager);
  }

#ifdef HAVE_HYPRE
  if (sim_comp == "solvertest2" || sim_comp == "SOLVERTEST2") {
    return scinew SolverTest2(myworld, materialManager);
  }
#endif

 if (sim_comp == "wave" || sim_comp == "WAVE") {
    if (doAMR) {
      return scinew AMRWave(myworld, materialManager);
    }
    else {
      return scinew Wave(myworld, materialManager);
    }
  }
  else {
    turned_on_options += "wave ";
  }

#if defined (HAVE_CUDA) && !defined(UINTAH_ENABLE_KOKKOS)
  if (sim_comp == "unifiedschedulertest" || sim_comp == "UNIFIEDSCHEDULERTEST") {
    return scinew UnifiedSchedulerTest(myworld, materialManager);
  }
  else {
    turned_on_options += "unifiedschedulertest ";
  }
#endif

#endif
  
  //----------------------------

  throw ProblemSetupException("ERROR<Application>: Unknown simulationComponent ('" + sim_comp + "'). It must one of the following: " + turned_on_options + "\n"
                              "Make sure that the requested application is supported in this build.", __FILE__, __LINE__);
}
