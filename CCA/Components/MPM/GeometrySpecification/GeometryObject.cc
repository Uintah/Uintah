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
     ps->require("toughness_min", d_toughnessMin);
     ps->require("toughness_max", d_toughnessMax);
     ps->require("toughness_variation", d_toughnessVariation);
   }
}

GeometryObject::~GeometryObject()
{
  delete d_piece;
}

IntVector GeometryObject::getNumParticlesPerCell()
{
  return d_resolution;
}

