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
#include <Packages/Uintah/CCA/Components/Switcher/Switcher.h>
#include <iosfwd>

using std::cerr;
using std::endl;

using namespace Uintah;

SimulationComponent* ComponentFactory::create(ProblemSpecP& ps, 
                                              const ProcessorGroup* world)
{
  string sim_comp = "";
  ps->get("SimulationComponent",sim_comp);

  SimulationInterface* sim = 0;
  UintahParallelComponent* comp = 0;
  SimulationComponent* d_simcomp = scinew SimulationComponent;

  if (sim_comp == "mpm" || sim_comp == "MPM") {
    SerialMPM* mpm = scinew SerialMPM(world);
    sim = mpm;
    comp = mpm;
  } else if (sim_comp == "fracturempm" || sim_comp == "FRACTUREMPM") {
    FractureMPM* fmpm = scinew FractureMPM(world);
    sim = fmpm;
    comp = fmpm;
  } else if (sim_comp == "rigidmpm" || sim_comp == "RIGIDMPM") {
    RigidMPM* rmpm = scinew RigidMPM(world);
    sim = rmpm;
    comp = rmpm;
  } else if (sim_comp == "shellmpm" || sim_comp == "SHELLMPM") {
    ShellMPM* smpm = scinew ShellMPM(world);
    sim = smpm;
    comp = smpm;
  } else if (sim_comp == "impm" || sim_comp == "IMPM") {
     ImpMPM* impm = scinew ImpMPM(world);
     sim = impm;
     comp = impm;
  } else if (sim_comp == "ice" || sim_comp == "ICE") {
    ICE* ice = scinew ICE(world);
    sim = ice;
    comp = ice;
  } else if (sim_comp == "amrice" || sim_comp == "AMRICE") {
    AMRICE* amrice = scinew AMRICE(world);
    sim = amrice;
    comp = amrice;
  } else if (sim_comp == "mpmice" || sim_comp == "MPMICE") {
    MPMICE* mpmice  = scinew MPMICE(world,STAND_MPMICE);
    sim = mpmice;
    comp = mpmice;
  } else if (sim_comp == "shellmpmice" || sim_comp == "SHELLMPMICE") {
    MPMICE* smpmice  = scinew MPMICE(world,SHELL_MPMICE);
    sim = smpmice;
    comp = smpmice;
  } else if (sim_comp == "rigidmpmice" || sim_comp == "RIGIDMPMICE") {
    MPMICE* rmpmice  = scinew MPMICE(world,RIGID_MPMICE);
    sim = rmpmice;
    comp = rmpmice;
  } else if (sim_comp == "fracturempmice" || sim_comp == "FRACTUREMPMICE") {
    MPMICE* fmpmice  = scinew MPMICE(world,FRACTURE_MPMICE);
    sim = fmpmice;
    comp = fmpmice;
  } else if (sim_comp == "mpmarches" || sim_comp == "MPMARCHES") {
    MPMArches* mpmarches  = scinew MPMArches(world);
    sim = mpmarches;
    comp = mpmarches;
  } else if (sim_comp == "burger" || sim_comp == "BURGER") {
    Burger* burger  = scinew Burger(world);
    sim = burger;
    comp = burger;
  } else if (sim_comp == "wave" || sim_comp == "WAVE") {
    Wave* wave  = scinew Wave(world);
    sim = wave;
    comp = wave;
  } else if (sim_comp == "amrwave" || sim_comp == "AMRWAVE") {
    Wave* amrwave  = scinew AMRWave(world);
    sim = amrwave;
    comp = amrwave;
  } else if (sim_comp == "poisson1" || sim_comp == "POISSON1") {
    Poisson1* poisson1  = scinew Poisson1(world);
    sim = poisson1;
    comp = poisson1;
  } else if (sim_comp == "regriddertest" || sim_comp == "REGRIDDERTEST") {
    RegridderTest* regriddertest  = scinew RegridderTest(world);
    sim = regriddertest;
    comp = regriddertest;
  } else if (sim_comp == "poisson2" || sim_comp == "POISSON2") {
    Poisson2* poisson2  = scinew Poisson2(world);
    sim = poisson2;
    comp = poisson2;
  } else if (sim_comp == "poisson3" || sim_comp == "POISSON3") {
    Poisson3* poisson3  = scinew Poisson3(world);
    sim = poisson3;
    comp = poisson3;
  } else if (sim_comp == "particletest" || sim_comp == "PARTICLETEST") {
    ParticleTest1* particletest  = scinew ParticleTest1(world);
    sim = particletest;
    comp = particletest;
  } else if (sim_comp == "solvertest" || sim_comp == "SOLVERTEST") {
    SolverTest1* solvertest  = scinew SolverTest1(world);
    sim = solvertest;
    comp = solvertest;
  } else if (sim_comp == "simplecfd" || sim_comp == "SIMPLECFD") {
    SimpleCFD* simplecfd  = scinew SimpleCFD(world);
    sim = simplecfd;
    comp = simplecfd;
  } else if (sim_comp == "amrsimplecfd" || sim_comp == "AMRSIMPLECFD") {
    AMRSimpleCFD* amrsimplecfd  = scinew AMRSimpleCFD(world);
    sim = amrsimplecfd;
    comp = amrsimplecfd;
  } else if (sim_comp == "test" || sim_comp == "TEST") {
    Test* test  = scinew Test(world);
    sim = test;
    comp = test;
  } else if (sim_comp == "switcher" || sim_comp == "SWITCHER") {
    Switcher* switcher  = scinew Switcher(world);
    sim = switcher;
    comp = switcher;
  } else {
    throw ProblemSetupException("Unknown simulationComponent");
  }
 
  d_simcomp->d_sim = sim;
  d_simcomp->d_comp = comp;

  return d_simcomp;

}
