#include "Fracture.h"

#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>

namespace Uintah {

Fracture::
Fracture(ProblemSpecP& ps)
{
  lb = scinew MPMLabel();
}

Fracture::~Fracture()
{
}

} // End namespace Uintah
