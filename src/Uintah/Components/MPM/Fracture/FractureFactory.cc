#include "FractureFactory.h"

#include "Fracture.h"

namespace Uintah {
namespace MPM {

Fracture* FractureFactory::create(const ProblemSpecP& ps, SimulationStateP &ss)
{

   ProblemSpecP mpm_ps = ps->findBlock("MaterialProperties")->findBlock("MPM");

   if(ProblemSpecP fracture = mpm_ps->findBlock("fracture"))
     return(new Fracture(fracture,ss));
   else 
     return 0;
}

} // end namespace MPM
} // end namespace Uintah


// $Log$
// Revision 1.1  2000/05/10 05:08:06  tan
// Basic structure of FractureFactory class.
//
