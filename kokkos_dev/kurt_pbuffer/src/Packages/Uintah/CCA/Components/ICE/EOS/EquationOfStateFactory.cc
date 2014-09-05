#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfStateFactory.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/IdealGas.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/JWL.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/JWLC.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/Murnahan.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/Gruneisen.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/Tillotson.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <string>
#if 0
#include <Packages/Uintah/CCA/Components/ICE/EOS/Harlow.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/StiffGas.h>
#endif

using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah;

EquationOfState* EquationOfStateFactory::create(ProblemSpecP& ps)
{
    ProblemSpecP child = ps->findBlock("EOS");
    if(!child)
      throw ProblemSetupException("Cannot find EOS tag");
    std::string mat_type;
    if(!child->getAttribute("type",mat_type))
      throw ProblemSetupException("No type for EOS"); 
    
    if (mat_type == "ideal_gas") 
      return(scinew IdealGas(child));
    else if (mat_type == "JWL") 
      return(scinew JWL(child));
    else if (mat_type == "JWLC") 
      return(scinew JWLC(child));
    else if (mat_type == "Murnahan") 
      return(scinew Murnahan(child));
    else if (mat_type == "Gruneisen") 
      return(scinew Gruneisen(child));
    else if (mat_type == "Tillotson") 
      return(scinew Tillotson(child));
#if 0   // Turn off harlow and stiff gas until everything with ideal
        // gas is working. Todd
    else if (mat_type == "harlow") 
      return(scinew Harlow(child));
    
    else if (mat_type == "stiff_gas") 
      return(scinew StiffGas(child));
#endif    
    else
      throw ProblemSetupException("Unknown EOS Type R ("+mat_type+")");

}
