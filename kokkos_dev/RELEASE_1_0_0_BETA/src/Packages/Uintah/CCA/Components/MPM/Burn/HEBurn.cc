
#include <Packages/Uintah/CCA/Components/MPM/Burn/HEBurn.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

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
       



