#include "FractureFactory.h"

#include "SimpleFracture.h"
#include "NormalFracture.h"
#include "ExplosiveFracture.h"

namespace Uintah {
namespace MPM {

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

} // end namespace MPM
} // end namespace Uintah


// $Log$
// Revision 1.4  2000/12/07 08:33:03  tan
// Compare the times for running the disk without fracture and with
// fracture (Simple fracture).
//
// Revision 1.3  2000/11/21 20:48:29  tan
// Implemented different models for fracture simulations.  SimpleFracture model
// is for the simulation where the resolution focus only on macroscopic major
// cracks. NormalFracture and ExplosionFracture models are more sophiscated
// and specific fracture models that are currently underconstruction.
//
// Revision 1.2  2000/09/05 05:13:55  tan
// Moved Fracture Model to MPMMaterial class.
//
// Revision 1.1  2000/05/10 05:08:06  tan
// Basic structure of FractureFactory class.
//
