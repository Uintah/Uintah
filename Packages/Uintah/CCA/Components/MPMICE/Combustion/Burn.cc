
#include <Packages/Uintah/CCA/Components/MPMICE/MPMICELabel.h>
#include <Packages/Uintah/CCA/Components/MPMICE/Combustion/Burn.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

Burn::Burn() 
{
  d_burnable = false;
  lb = scinew MPMICELabel();
}

Burn::~Burn()
{
  delete lb;
}

bool Burn::isBurnable() 
{
   return d_burnable;
}
       



