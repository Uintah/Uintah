#include "Fracture.h"

#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>

namespace Uintah {

Fracture::Fracture(ProblemSpecP& ps)
{
  lb = scinew MPMLabel();

  ps->require("pressure_rate",d_pressureRate);
  ps->require("constraint",d_constraint);
}

Fracture::~Fracture()
{
}

double Fracture::getPressureRate() const
{
  return d_pressureRate;
}

int Fracture::getConstraint() const
{
  return d_constraint;
}

} // End namespace Uintah
