#include "FractureFactory.h"

#include "NormalFracture.h"

namespace Uintah {
Fracture* FractureFactory::create(const ProblemSpecP& ps)
{
   if(ProblemSpecP fractureProb = ps->findBlock("fracture")) {
     std::string fracture_type;
     fractureProb->require("type",fracture_type);
     cerr << "fracture_type = " << fracture_type << std::endl;
 
     if (fracture_type == "normal") 
        return(scinew NormalFracture(fractureProb));
     else {
	cerr << "Unknown Fracture Type (" << fracture_type << ")" << std::endl;
     }
   }
   return NULL;
}
} // End namespace Uintah



