#include "Fracture.h"

#include <Uintah/Components/MPM/ConstitutiveModel/MPMMaterial.h>

#include <Uintah/Components/MPM/MPMLabel.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/NCVariable.h>

#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Grid/CellIterator.h>

#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Geometry/Point.h>

namespace Uintah {
namespace MPM {

using SCICore::Geometry::Vector;
using SCICore::Geometry::Point;

Fracture::CellStatus
Fracture::
cellStatus(const Vector& cellSurfaceNormal)
{
  if(cellSurfaceNormal.x() > 1000) return HAS_SEVERAL_BOUNDARY_SURFACE;
  else if( fabs(cellSurfaceNormal.x()) +
           fabs(cellSurfaceNormal.y()) +
           fabs(cellSurfaceNormal.z()) < 0.1 ) return INTERIOR_CELL;
  else return HAS_ONE_BOUNDARY_SURFACE;
} 

void
Fracture::
setCellStatus(Fracture::CellStatus status,Vector* cellSurfaceNormal)
{
  if(status == HAS_SEVERAL_BOUNDARY_SURFACE) cellSurfaceNormal->x(1000.1);
  else if(status == INTERIOR_CELL) (*cellSurfaceNormal) = Vector(0.,0.,0.);
}
 
Fracture::ParticleStatus
Fracture::
particleStatus(const Vector& particleSurfaceNormal)
{
  if( fabs(particleSurfaceNormal.x()) +
      fabs(particleSurfaceNormal.y()) +
      fabs(particleSurfaceNormal.z()) < 0.1 ) return INTERIOR_PARTICLE;
  else return BOUNDARY_PARTICLE;
}

void
Fracture::
setParticleStatus(Fracture::ParticleStatus status,Vector* particleSurfaceNormal)
{
  if(status == INTERIOR_PARTICLE) (*particleSurfaceNormal) = Vector(0.,0.,0.);
}

void
Fracture::
materialDefectsInitialize(const Patch* patch,
                          DataWarehouseP& new_dw)
{
}

void
Fracture::
initializeFracture(const Patch* patch,
                  DataWarehouseP& new_dw)
{
  int vfindex = d_sharedState->getMaterial(0)->getVFIndex();

  const MPMLabel* lb = MPMLabel::getLabels();
  
  //For CCVariables
  //set default cSelfContact to false
  CCVariable<bool> cSelfContact;
  new_dw->allocate(cSelfContact, lb->cSelfContactLabel, vfindex, patch);

  //set default cSurfaceNormal to [0.,0.,0.]
  CCVariable<Vector> cSurfaceNormal;
  new_dw->allocate(cSurfaceNormal, lb->cSurfaceNormalLabel, vfindex, patch);
  Vector zero(0.,0.,0.);

  for(CellIterator iter = patch->getCellIterator(patch->getBox());
                   !iter.done(); 
                   iter++)
  {
    cSelfContact[*iter] = false;
    cSurfaceNormal[*iter] = zero;
  }

  //For NCVariables
  NCVariable<bool> gSelfContact;
  new_dw->allocate(gSelfContact, lb->gSelfContactLabel, vfindex, patch);

  for(NodeIterator iter = patch->getNodeIterator();
                   !iter.done(); 
                   iter++)
  {
    gSelfContact[*iter] = false;
  }



  //put back to DatawareHouse
  new_dw->put(cSelfContact, lb->cSelfContactLabel, vfindex, patch);
  new_dw->put(cSurfaceNormal, lb->cSurfaceNormalLabel, vfindex, patch);
  new_dw->put(gSelfContact, lb->gSelfContactLabel, vfindex, patch);

  materialDefectsInitialize(patch, new_dw);
}

void
Fracture::
labelCellSurfaceNormal (
           const ProcessorContext*,
           const Patch* patch,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMatls();
  const MPMLabel* lb = MPMLabel::getLabels();

  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int matlindex = matl->getDWIndex();
      int vfindex = matl->getVFIndex();

      ParticleVariable<Point> px;
      ParticleVariable<Vector> pSurfaceNormal;

      old_dw->get(px, lb->pXLabel, matlindex, patch,
		  Ghost::None, 0);
      old_dw->get(pSurfaceNormal, lb->pSurfaceNormalLabel, matlindex, patch,
		  Ghost::None, 0);

      CCVariable<Vector> cSurfaceNormal;
      new_dw->allocate(cSurfaceNormal, lb->cSurfaceNormalLabel, vfindex, patch);

      ParticleSubset* pset = px.getParticleSubset();
      cSurfaceNormal.initialize(Vector(0,0,0));

      for(ParticleSubset::iterator part_iter = pset->begin();
	  part_iter != pset->end(); part_iter++)
      {
	 particleIndex pIdx = *part_iter;
	 IntVector cIdx = patch->findCell(px[pIdx]);

	 Vector cellSurfaceNormal = cSurfaceNormal[cIdx];
	 if( cellStatus(cellSurfaceNormal) == HAS_SEVERAL_BOUNDARY_SURFACE)
	   continue;
	 
	 Vector particleSurfaceNormal = pSurfaceNormal[pIdx];
	 if( SCICore::Geometry::Dot(cellSurfaceNormal,particleSurfaceNormal) > 0 ) {
	   cSurfaceNormal[cIdx] += particleSurfaceNormal;
	 }
      };

      for(CellIterator cell_iter = patch->getCellIterator(patch->getBox()); 
        !cell_iter.done(); cell_iter++)
      {
        if( ( cellStatus(cSurfaceNormal[*cell_iter]) == 
              HAS_SEVERAL_BOUNDARY_SURFACE ) ||
            ( cellStatus(cSurfaceNormal[*cell_iter]) == 
              INTERIOR_CELL) )
        {
          cSurfaceNormal[*cell_iter].normalize();
        }
      }
      
      new_dw->put(cSurfaceNormal, lb->cSurfaceNormalLabel, vfindex, patch);
    }
  }
}

void
Fracture::
labelSelfContactNodes(
           const ProcessorContext*,
           const Patch* patch,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw)
{
  int numMatls = d_sharedState->getNumMatls();

  const MPMLabel* lb = MPMLabel::getLabels();

  for(int m = 0; m < numMatls; m++){
    Material* matl = d_sharedState->getMaterial( m );
    MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
    if(mpm_matl){
      int matlindex = matl->getDWIndex();
      int vfindex = matl->getVFIndex();

      CCVariable<Vector> cSurfaceNormal;
      new_dw->get(cSurfaceNormal, lb->cSurfaceNormalLabel, matlindex, patch,
		  Ghost::AroundNodes, 1);
		  
      NCVariable<bool> gSelfContact;
      new_dw->allocate(gSelfContact, lb->gSelfContactLabel, vfindex, patch);
      
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++)
      {
        gSelfContact[*iter] = isSelfContactNode(*iter,patch,cSurfaceNormal);
      }

      new_dw->put(gSelfContact, lb->gSelfContactLabel, vfindex, patch);
    }
  }
}

bool
Fracture::
isSelfContactNode(const IntVector& nodeIndex,const Patch* patch,
  const CCVariable<Vector>& cSurfaceNormal)
{
  IntVector cellIndex[8];
  patch->findCellsFromNode(nodeIndex,cellIndex);
        
  for(int k = 0; k < 8; k++) {
    if( cellStatus(cSurfaceNormal[cellIndex[k]]) == 
	    HAS_SEVERAL_BOUNDARY_SURFACE ) return true;
  }
  return false;
}

void
Fracture::
labelSelfContactNodesAndCells(
           const ProcessorContext*,
           const Patch* patch,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw)
{
#if 0
  int vfindex = d_sharedState->getMaterial(0)->getVFIndex();

  CCVariable<bool> cSelfContact;
  CCVariable<Vector> cSurfaceNormal;

  new_dw->allocate(cSelfContact, cSelfContactLabel, vfindex, region);
  new_dw->allocate(cSurfaceNormal cSurfaceNormalLabel, vfindex, region);

  //Label out the cells having non-directly connected boundary particles
  //The information is saved temperary in cSelfContact

  //Label out the self-contact nodes

  for(NodeIterator nodeIter = region->getNodeIterator();
                   !nodeIter.done(); 
                   nodeIter++)
  {
    cSelfContact[*cellIter] = false;
  }

  //Label out the self-contact cells

  for(CellIterator cellIter = region->getCellIterator();
                   !cellIter.done(); 
                   cellIter++)
  {
    cSelfContact[*cellIter] = false;
  }
#endif

}

void
Fracture::
updateParticleInformationInContactCells (
           const ProcessorContext*,
           const Patch* patch,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw)
{
}

void
Fracture::
updateSurfaceNormalOfBoundaryParticle(
	   const ProcessorContext*,
           const Patch* patch,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw)
{
   // Added empty function - Steve
}
  
void
Fracture::
updateNodeInformationInContactCells (
           const ProcessorContext*,
           const Patch* patch,
           DataWarehouseP& old_dw,
           DataWarehouseP& new_dw)
{
}


void
Fracture::
crackGrow(
           const ProcessorContext*,
           const Patch* patch,
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
// Revision 1.17  2000/06/02 21:54:22  tan
// Finished function labelSelfContactNodes(...) to label the gSalfContact
// according to the cSurfaceNormal information.
//
// Revision 1.16  2000/06/02 21:12:24  tan
// Added function isSelfContactNode(...) to determine if a node is a
// self-contact node.
//
// Revision 1.15  2000/06/02 19:13:39  tan
// Finished function labelCellSurfaceNormal() to label the cell surface normal
// according to the boundary particles surface normal information.
//
// Revision 1.14  2000/06/02 00:13:13  tan
// Added ParticleStatus to determine if a particle is a BOUNDARY_PARTICLE
// or a INTERIOR_PARTICLE.
//
// Revision 1.13  2000/06/01 23:56:00  tan
// Added CellStatus to determine if a cell HAS_ONE_BOUNDARY_SURFACE,
// HAS_SEVERAL_BOUNDARY_SURFACE or is INTERIOR cell.
//
// Revision 1.12  2000/05/30 20:19:12  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
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
