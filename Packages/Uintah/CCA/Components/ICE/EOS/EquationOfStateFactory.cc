#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfStateFactory.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/IdealGas.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/HardSphereGas.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/JWL.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/TST.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/JWLC.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/Murnahan.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/Gruneisen.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/Tillotson.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/Thomsen_Hartka_water.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <string>

using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah;

EquationOfState* EquationOfStateFactory::create(ProblemSpecP& ps)
{
  ProblemSpecP EOS_ps = ps->findBlock("EOS");
  if(!EOS_ps){
    throw ProblemSetupException("ERROR ICE: Cannot find EOS tag", __FILE__, __LINE__);
  }
  
  std::string EOS;
  if(!EOS_ps->getAttribute("type",EOS)){
    throw ProblemSetupException("ERROR ICE: Cannot find EOS 'type' tag", __FILE__, __LINE__); 
  }
  if (EOS == "ideal_gas") 
    return(scinew IdealGas(EOS_ps));
  else if (EOS == "hard_sphere_gas") 
    return(scinew HardSphereGas(EOS_ps));
  else if (EOS == "TST") 
    return(scinew TST(EOS_ps));
  else if (EOS == "JWL") 
    return(scinew JWL(EOS_ps));
  else if (EOS == "JWLC") 
    return(scinew JWLC(EOS_ps));
  else if (EOS == "Murnahan") 
    return(scinew Murnahan(EOS_ps));
  else if (EOS == "Gruneisen") 
    return(scinew Gruneisen(EOS_ps));
  else if (EOS == "Tillotson") 
    return(scinew Tillotson(EOS_ps));    
  else if (EOS == "Thomsen_Hartka_water") 
    return(scinew Thomsen_Hartka_water(EOS_ps));    
  else{
    ostringstream warn;
    warn << "ERROR ICE: Unknown Equation of State ("<< EOS << " )\n"
         << "Valid equations of State:\n" 
         << "ideal_gas\n"
         << "TST\n"
         << "JWL\n"
         << "JWLC\n"
         << "Murnahan\n"
         << "Gruneisen\n"
         << "Tillotson\n"
         << "Thomsen_Hartka_water" << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

}
