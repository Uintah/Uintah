#include "FractureFactory.h"

#include "SimpleFracture.h"
#include "NormalFracture.h"
#include "ExplosiveFracture.h"

namespace Uintah {
Fracture* FractureFactory::create(const ProblemSpecP& ps)
{
   if(ProblemSpecP fractureProb = ps->findBlock("fracture")) {
     std::string fracture_type;
     fractureProb->require("type",fracture_type);
     cerr << "fracture_type = " << fracture_type << std::endl;
 
     if (fracture_type == "normal") 
        return(scinew NormalFracture(fractureProb));
     else if (fracture_type == "simple") 
        return(scinew SimpleFracture(fractureProb));
     else if (fracture_type == "explosive") 
        return(scinew ExplosiveFracture(fractureProb));
     else {
	cerr << "Unknown Fracture Type (" << fracture_type << ")" << std::endl;
     }
   }
   return NULL;
}
} // End namespace Uintah



