
#include <Uintah/Components/MPM/Burn/HEBurn.h>
#include <SCICore/Malloc/Allocator.h>

using namespace Uintah::MPM;

HEBurn::HEBurn() 
{
  d_burnable = false;
  lb = scinew MPMLabel();
}

HEBurn::~HEBurn()
{
  delete lb;
}

bool HEBurn::isBurnable() 
{
   return d_burnable;
}
       

// $Log$
// Revision 1.3  2000/08/09 03:17:59  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.2  2000/07/05 23:43:31  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.1  2000/06/02 22:48:25  jas
// Added infrastructure for Burn models.
//


