#include "Fracture.h"

#include <Uintah/Components/MPM/MPMLabel.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>

#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/Region.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/CellIterator.h>

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
                          DataWarehouseP& new_dw)
{
}

void
Fracture::
initializeFracture(const Region* region,
                  DataWarehouseP& new_dw)
{
  int vfindex = d_sharedState->getMaterial(0)->getVFIndex();

  const MPMLabel* lb = MPMLabel::getLabels();
  
  //For CCVariables
  //set default cSelfContact to false
  CCVariable<bool> cSelfContact;
  new_dw->allocate(cSelfContact, lb->cSelfContactLabel, vfindex, region);

  //set default cSurfaceNormal to [0.,0.,0.]
  CCVariable<Vector> cSurfaceNormal;
  new_dw->allocate(cSurfaceNormal, lb->cSurfaceNormalLabel, vfindex, region);
  Vector zero(0.,0.,0.);

  for(CellIterator iter = region->getCellIterator(region->getBox());
                   !iter.done(); 
                   iter++)
  {
    cSelfContact[*iter] = false;
    cSurfaceNormal[*iter] = zero;
  }

  //For NCVariables
  NCVariable<bool> gSelfContact;
  new_dw->allocate(gSelfContact, lb->gSelfContactLabel, vfindex, region);

  for(NodeIterator iter = region->getNodeIterator();
                   !iter.done(); 
                   iter++)
  {
    gSelfContact[*iter] = false;
  }



  //put back to DatawareHouse
  new_dw->put(cSelfContact, lb->cSelfContactLabel, vfindex, region);
  new_dw->put(cSurfaceNormal, lb->cSurfaceNormalLabel, vfindex, region);
  new_dw->put(gSelfContact, lb->gSelfContactLabel, vfindex, region);

  materialDefectsInitialize(region, new_dw);
}

void
Fracture::
labelSelfContactNodesAndCells(
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
updateSurfaceNormalOfBoundaryParticle(
	   const ProcessorContext*,
           const Region* region,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw)
{
   // Added empty function - Steve
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
}
  
} //namespace MPM
} //namespace Uintah

// $Log$
// Revision 1.11  2000/05/30 04:37:00  tan
// Using MPMLabel instead of VarLabel.
//
// Revision 1.10  2000/05/25 00:29:00  tan
// Put all velocity-field independent variables on material index of 0.
//
// Revision 1.9  2000/05/20 08:09:09  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.8  2000/05/15 18:59:10  tan
// Initialized NCVariables and CCVaribles for Fracture.
//
// Revision 1.7  2000/05/12 18:13:07  sparker
// Added an empty function for Fracture::updateSurfaceNormalOfBoundaryParticle
//
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
