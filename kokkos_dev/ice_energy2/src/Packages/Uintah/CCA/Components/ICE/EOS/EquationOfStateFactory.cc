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
#include <string>
#if 0
#include <Packages/Uintah/CCA/Components/ICE/EOS/Harlow.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/StiffGas.h>
#endif

using namespace Uintah;

EquationOfState* EquationOfStateFactory::create(ProblemSpecP& ps, ICEMaterial* ice_matl)
{
    ProblemSpecP child = ps->findBlock("EOS");
    if(!child)
      throw ProblemSetupException("Cannot find EOS tag", __FILE__, __LINE__);
    std::string mat_type;
    if(!child->getAttribute("type",mat_type))
      throw ProblemSetupException("No type for EOS", __FILE__, __LINE__); 
    
    if (mat_type == "ideal_gas") 
      return(scinew IdealGas(child, ice_matl));
#if 0
    // Others busted - Steve
    else if (mat_type == "JWL") 
      return(scinew JWL(child, ice_matl));
    else if (mat_type == "JWLC") 
      return(scinew JWLC(child, ice_matl));
    else if (mat_type == "Murnahan") 
      return(scinew Murnahan(child, ice_matl));
    else if (mat_type == "Gruneisen") 
      return(scinew Gruneisen(child, ice_matl));
    else if (mat_type == "Tillotson") 
      return(scinew Tillotson(child, ice_matl));
#if 0   // Turn off harlow and stiff gas until everything with ideal
        // gas is working. Todd
    else if (mat_type == "harlow") 
      return(scinew Harlow(child));
    
    else if (mat_type == "stiff_gas") 
      return(scinew StiffGas(child));
#endif    
#endif
    else
      throw ProblemSetupException("Unknown EOS Type R ("+mat_type+")", __FILE__, __LINE__);

}
