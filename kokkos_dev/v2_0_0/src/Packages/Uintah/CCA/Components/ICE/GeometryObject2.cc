#include <Packages/Uintah/CCA/Components/ICE/GeometryObject2.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>

using namespace Uintah;

GeometryObject2::GeometryObject2(ICEMaterial* /*ice_matl*/,
                               GeometryPiece* piece,
                            ProblemSpecP& ps)
   : d_piece(piece)
{
   ps->require("res",         d_resolution);
   ps->require("velocity",    d_initialVel);
   ps->require("temperature", d_initialTemperature);
   ps->require("pressure",    d_initialPressure);
   ps->require("density",     d_initialDensity);
}

GeometryObject2::~GeometryObject2()
{
}

IntVector GeometryObject2::getNumParticlesPerCell()
{
  return d_resolution;
}

