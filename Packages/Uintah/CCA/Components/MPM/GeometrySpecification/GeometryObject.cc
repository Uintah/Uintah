#include "GeometryObject.h"
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <iostream>

using namespace Uintah;

GeometryObject::GeometryObject(MPMMaterial* mpm_matl,
                               GeometryPiece* piece,
			       ProblemSpecP& ps)
   : d_piece(piece)
{
   ps->require("res", d_resolution);
   ps->require("velocity", d_initialVel);
   ps->require("temperature", d_initialTemperature);
   
   if(mpm_matl->getFractureModel()) {
     ps->require("tensile_strength_min", d_tensileStrengthMin);
     ps->require("tensile_strength_max", d_tensileStrengthMax);
     ps->require("tensile_strength_variation", d_tensileStrengthVariation);
   }
}

GeometryObject::~GeometryObject()
{
}

IntVector GeometryObject::getNumParticlesPerCell()
{
  return d_resolution;
}

