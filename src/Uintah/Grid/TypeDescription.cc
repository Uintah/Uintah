
#include <Uintah/Grid/TypeDescription.h>

using namespace Uintah;

TypeDescription::TypeDescription(bool reductionvar, Basis basis)
   : d_reductionvar(reductionvar), d_basis(basis)
{
}

//
// $Log$
// Revision 1.1  2000/05/07 06:02:13  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
//

