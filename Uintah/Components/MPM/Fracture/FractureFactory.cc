#include "FractureFactory.h"

#include "Fracture.h"

namespace Uintah {
namespace MPM {

Fracture* FractureFactory::create(const ProblemSpecP& ps)
{
   if(ProblemSpecP fractureProb = ps->findBlock("fracture"))
     return(new Fracture(fractureProb));
   else 
     return NULL;
}

} // end namespace MPM
} // end namespace Uintah


// $Log$
// Revision 1.2  2000/09/05 05:13:55  tan
// Moved Fracture Model to MPMMaterial class.
//
// Revision 1.1  2000/05/10 05:08:06  tan
// Basic structure of FractureFactory class.
//
