#include "FractureFactory.h"
#include "NormalFracture.h"
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

namespace Uintah {
Fracture* FractureFactory::create(const ProblemSpecP& ps)
{
   ProblemSpecP fractureProb = ps->findBlock("fracture");
   if (fractureProb) {
     std::string fracture_type;
     if (!fractureProb->getAttribute("type",fracture_type))
       throw ProblemSetupException("No type for fracture");

     if (fracture_type == "normal") 
        return(scinew NormalFracture(fractureProb));

     else 
	throw ProblemSetupException("Unknown Fracture Type ("+fracture_type+")");
     
   }
   return 0;
}

} // End namespace Uintah



