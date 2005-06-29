#include <Packages/Uintah/CCA/Components/ComponentFactory.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/CCA/Components/MPM/SerialMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/FractureMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/RigidMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/ShellMPM.h>
#include <Packages/Uintah/CCA/Components/MPM/ImpMPM.h>
#include <Packages/Uintah/CCA/Components/ICE/ICE.h>
#include <Packages/Uintah/CCA/Components/ICE/AMRICE.h>
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICE.h>
#include <Packages/Uintah/CCA/Components/MPMArches/MPMArches.h>
#include <Packages/Uintah/CCA/Components/Examples/Poisson1.h>
#include <Packages/Uintah/CCA/Components/Examples/Poisson2.h>
#include <Packages/Uintah/CCA/Components/Examples/Burger.h>
#include <Packages/Uintah/CCA/Components/Examples/Wave.h>
#include <Packages/Uintah/CCA/Components/Examples/AMRWave.h>
#include <Packages/Uintah/CCA/Components/Examples/ParticleTest1.h>
#include <Packages/Uintah/CCA/Components/Examples/RegridderTest.h>
#include <Packages/Uintah/CCA/Components/Examples/Poisson3.h>
#include <Packages/Uintah/CCA/Components/Examples/SimpleCFD.h>
#include <Packages/Uintah/CCA/Components/Examples/AMRSimpleCFD.h>
#include <Packages/Uintah/CCA/Components/Examples/SolverTest1.h>
#include <Packages/Uintah/CCA/Components/Examples/Test.h>
#include <Packages/Uintah/CCA/Components/Examples/Test1.h>
#include <Packages/Uintah/CCA/Components/Switcher/Switcher.h>
#include <iosfwd>

using std::cerr;
using std::endl;

using namespace Uintah;

UintahParallelComponent* ComponentFactory::create(ProblemSpecP& ps, const ProcessorGroup* world, bool doAMR)
{
  string sim_comp = "";

  ProblemSpecP sim_ps = ps->findBlock("SimulationComponent");
  if (sim_ps)
    sim_ps->get("type",sim_comp);

  if (sim_comp == "mpm" || sim_comp == "MPM") {
    return scinew SerialMPM(world);
  } else if (sim_comp == "fracturempm" || sim_comp == "FRACTUREMPM") {
    return scinew FractureMPM(world);
  } else if (sim_comp == "rigidmpm" || sim_comp == "RIGIDMPM") {
    return scinew RigidMPM(world);
  } else if (sim_comp == "shellmpm" || sim_comp == "SHELLMPM") {
    return scinew ShellMPM(world);
  } else if (sim_comp == "impm" || sim_comp == "IMPM") {
    return scinew ImpMPM(world);
  } else if (sim_comp == "ice" || sim_comp == "ICE") {
    if (doAMR)
      return scinew AMRICE(world);
    else
      return scinew ICE(world);
  } else if (sim_comp == "mpmice" || sim_comp == "MPMICE") {
    return scinew MPMICE(world,STAND_MPMICE, doAMR);
  } else if (sim_comp == "shellmpmice" || sim_comp == "SHELLMPMICE") {
    return scinew MPMICE(world,SHELL_MPMICE, doAMR);
  } else if (sim_comp == "rigidmpmice" || sim_comp == "RIGIDMPMICE") {
    return scinew MPMICE(world,RIGID_MPMICE, doAMR);
  } else if (sim_comp == "fracturempmice" || sim_comp == "FRACTUREMPMICE") {
    return scinew MPMICE(world,FRACTURE_MPMICE, doAMR);
  } else if (sim_comp == "mpmarches" || sim_comp == "MPMARCHES") {
    return scinew MPMArches(world);
  } else if (sim_comp == "burger" || sim_comp == "BURGER") {
    return scinew Burger(world);
  } else if (sim_comp == "wave" || sim_comp == "WAVE") {
    if (doAMR)
      return scinew AMRWave(world);
    else
      return scinew Wave(world);
  } else if (sim_comp == "poisson1" || sim_comp == "POISSON1") {
    return scinew Poisson1(world);
  } else if (sim_comp == "regriddertest" || sim_comp == "REGRIDDERTEST") {
    return scinew RegridderTest(world);
  } else if (sim_comp == "poisson2" || sim_comp == "POISSON2") {
    return scinew Poisson2(world);
  } else if (sim_comp == "poisson3" || sim_comp == "POISSON3") {
    return scinew Poisson3(world);
  } else if (sim_comp == "particletest" || sim_comp == "PARTICLETEST") {
    return scinew ParticleTest1(world);
  } else if (sim_comp == "solvertest" || sim_comp == "SOLVERTEST") {
    return scinew SolverTest1(world);
  } else if (sim_comp == "simplecfd" || sim_comp == "SIMPLECFD") {
    if (doAMR)
      return scinew AMRSimpleCFD(world);
    else
      return scinew SimpleCFD(world);
  } else if (sim_comp == "test" || sim_comp == "TEST") {
    return scinew Test(world);
  } else if (sim_comp == "test1" || sim_comp == "TEST1") {
    return scinew Test1(world);
  } else if (sim_comp == "switcher" || sim_comp == "SWITCHER") {
    return scinew Switcher(world, ps, doAMR);
  } else {
    char *c = 0;
    *c = 0;
    throw ProblemSetupException("Unknown simulationComponent. Must specify -arches, -ice, -mpm, "
		  "-impm, -fmpmice, -mpmice, -mpmarches, -burger, -wave, -poisson1, -poisson2, or -poisson3");
  }
 
}
