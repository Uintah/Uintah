
#include <Uintah/Grid/TypeDescription.h>

using namespace Uintah;

TypeDescription::TypeDescription(bool reductionvar, Basis basis, Type type)
   : d_reductionvar(reductionvar), d_basis(basis), d_type(type)
{
}


//
// $Log$
// Revision 1.2  2000/05/18 18:41:14  kuzimmer
// Added Particle to Basis enum, created Type enum with Scalar,Point,Vector,Tensor,& Other
//
// Revision 1.1  2000/05/07 06:02:13  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
//

