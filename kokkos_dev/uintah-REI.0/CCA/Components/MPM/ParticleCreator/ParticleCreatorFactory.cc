#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ParticleCreatorFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ImplicitParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/DefaultParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/MembraneParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/ShellParticleCreator.h>
#include <Packages/Uintah/CCA/Components/MPM/ParticleCreator/FractureParticleCreator.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Core/Malloc/Allocator.h>
#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sgi_stl_warnings_on.h>
using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah;

ParticleCreator* ParticleCreatorFactory::create(ProblemSpecP& ps, 
                                                MPMMaterial* mat,
                                                MPMFlags* flags)
{

  ProblemSpecP cm_ps = ps->findBlock("constitutive_model");
  string mat_type;
  cm_ps->getAttribute("type",mat_type);

  if (flags->d_integrator_type == "implicit") 
    return scinew ImplicitParticleCreator(mat,flags);

  else if (flags->d_integrator_type == "fracture") 
    return scinew FractureParticleCreator(mat,flags);

  else if (mat_type == "membrane")
    return scinew MembraneParticleCreator(mat,flags);

  else if (mat_type == "shell_CNH")
    return scinew ShellParticleCreator(mat,flags);
  
  else
    return scinew DefaultParticleCreator(mat,flags);

}


