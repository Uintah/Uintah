#include "Fracture.h"

#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/CCVariable.h>

#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

namespace Uintah {
namespace MPM {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;
 
void
Fracture::
materialDefectsInitialize()
{
}

void
Fracture::
updateSurfaceNormalOfBoundaryParticle(
           const ProcessorContext*,
           const Region* region,
           const DataWarehouseP& old_dw,
           DataWarehouseP& new_dw)
{
}

void
Fracture::
crackGrow(
           const ProcessorContext*,
           const Region* region,
           const DataWarehouseP& old_dw,
           DataWarehouseP& new_dw)
{
}

Fracture::
Fracture(ProblemSpecP& ps,SimulationStateP& d_sS)
{
  ps->require("averageMicrocrackLength",d_averageMicrocrackLength);
  ps->require("materialToughness",d_materialToughness);

  d_sharedState = d_sS;

  pSurfaceNormalLabel = 
    new VarLabel( "p.surfaceNormal",
                   ParticleVariable<Vector>::getTypeDescription() );

  pStressLabel   = new VarLabel( "p.stress",
                   ParticleVariable<Matrix3>::getTypeDescription() );

  pDeformationMeasureLabel = new VarLabel("p.deformationMeasure",
                             ParticleVariable<Matrix3>::getTypeDescription());

  pXLabel        = new VarLabel( "p.x",
	           ParticleVariable<Point>::getTypeDescription(),
                   VarLabel::PositionVariable);

  cSelfContactLabel = new VarLabel( "c.selfContact",
                      CCVariable<Matrix3>::getTypeDescription() );
};
  
} //namespace MPM
} //namespace Uintah

// $Log$
// Revision 1.3  2000/05/10 05:06:40  tan
// Basic structure of fracture class.
//
