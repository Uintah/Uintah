
#include <Uintah/Components/MPM/Burn/HEBurn.h>
#include <SCICore/Malloc/Allocator.h>

using namespace Uintah::MPM;

HEBurn::HEBurn() 
{
  d_burnable = false;
}

bool HEBurn::isBurnable() 
{
   return d_burnable;
}
	

// $Log$
// Revision 1.1  2000/06/02 22:48:25  jas
// Added infrastructure for Burn models.
//


