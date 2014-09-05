#include "Fracture.h"

#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>

namespace Uintah {

Fracture::Fracture(ProblemSpecP& ps)
{
  lb = scinew MPMLabel();

  ps->require("constraint",d_constraint);
}

Fracture::~Fracture()
{
}

int Fracture::getConstraint() const
{
  return d_constraint;
}

} // End namespace Uintah
