#include "Fracture.h"

#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Grid/NodeIterator.h>

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
materialDefectsInitialize(const Region* region,
                          int vfindex,
                          DataWarehouseP& new_dw)
{
}

void
Fracture::
initializeFracture(const Region* region,
                  int vfindex,
                  DataWarehouseP& new_dw)
{
  //set default c.selfContact to false
  CCVariable<bool> selfContact;
  new_dw->allocate(selfContact, cSelfContactLabel, vfindex, region);
  selfContact.initialize(false);
  new_dw->put(selfContact,cSelfContactLabel,vfindex,region);

  
  //set default c.surfaceNormal to [0.,0.,0.]
  CCVariable<Vector> surfaceNormal;
  new_dw->allocate(surfaceNormal, cSurfaceNormalLabel, vfindex, region);
  surfaceNormal.initialize(Vector(0.0,0.0,0.0));
  new_dw->put(surfaceNormal, cSurfaceNormalLabel, vfindex, region);

  materialDefectsInitialize(region, vfindex, new_dw);
}

void
Fracture::
labelSelfContactCells (
           const ProcessorContext*,
           const Region* region,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw)
{
}

void
Fracture::
updateParticleInformationInContactCells (
           const ProcessorContext*,
           const Region* region,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw)
{
}

void
Fracture::
updateNodeInformationInContactCells (
           const ProcessorContext*,
           const Region* region,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw)
{
}


void
Fracture::
crackGrow(
           const ProcessorContext*,
           const Region* region,
           DataWarehouseP& old_dw,
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

  pExternalForceLabel = new VarLabel( "p.externalForce",
                   ParticleVariable<Vector>::getTypeDescription() );

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
// Revision 1.6  2000/05/12 01:46:07  tan
// Added initializeFracture linked to SerialMPM's actuallyInitailize.
//
// Revision 1.5  2000/05/11 20:10:18  dav
// adding MPI stuff.  The biggest change is that old_dws cannot be const and so a large number of declarations had to change.
//
// Revision 1.4  2000/05/10 18:32:11  tan
// Added member funtion to label self-contact cells.
//
// Revision 1.3  2000/05/10 05:06:40  tan
// Basic structure of fracture class.
//
