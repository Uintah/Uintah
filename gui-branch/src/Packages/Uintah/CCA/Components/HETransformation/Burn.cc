
#include <Packages/Uintah/CCA/Components/HETransformation/Burn.h>

using namespace Uintah;

Burn::Burn() 
{
  d_burnable = false;
}

Burn::~Burn()
{
}

bool Burn::isBurnable() 
{
   return d_burnable;
}
       



