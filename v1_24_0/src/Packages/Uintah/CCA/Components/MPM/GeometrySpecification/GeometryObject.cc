#include <Packages/Uintah/CCA/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;

GeometryObject::GeometryObject(MPMMaterial*,
                               GeometryPiece* piece,
			       ProblemSpecP& ps)
   : d_piece(piece)
{
   ps->require("res", d_resolution);
   ps->require("velocity", d_initialVel);
   ps->require("temperature", d_initialTemperature);
}

GeometryObject::~GeometryObject()
{
  delete d_piece;
}

IntVector GeometryObject::getNumParticlesPerCell()
{
  return d_resolution;
}

