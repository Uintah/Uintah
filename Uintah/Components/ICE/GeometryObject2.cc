#include "GeometryObject2.h"
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Components/ICE/ICEMaterial.h>
#include <Uintah/Grid/GeometryPiece.h>
#include <iostream>

using namespace Uintah::ICESpace;
using namespace Uintah;

GeometryObject2::GeometryObject2(ICEMaterial* ice_matl,
                               GeometryPiece* piece,
			       ProblemSpecP& ps)
   : d_piece(piece)
{
   ps->require("res", d_resolution);
   ps->require("velocity", d_initialVel);
   ps->require("temperature", d_initialTemperature);
}

GeometryObject2::~GeometryObject2()
{
}

// $Log$
// Revision 1.1  2000/11/22 01:28:05  guilkey
// Changed the way initial conditions are set.  GeometryObjects are created
// to fill the volume of the domain.  Each object has appropriate initial
// conditions associated with it.  ICEMaterial now has an initializeCells
// method, which for now just does what was previously done with the
// initial condition stuct d_ic.  This will be extended to allow regions of
// the domain to be initialized with different materials.  Sorry for the
// lame GeometryObject2, this could be changed to ICEGeometryObject or
// something.
//
