#include "ParticleCreator.h"
#include <Packages/Uintah/CCA/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/Box.h>

using namespace Uintah;

ParticleCreator::ParticleCreator()
{
}

ParticleCreator::~ParticleCreator()
{
}


void ParticleCreator::createParticles(MPMMaterial* matl,
				      particleIndex numParticles,
				      CCVariable<short int>& cellNAPID,
				      const Patch*,DataWarehouse* new_dw,
				      MPMLabel* lb,
				      std::vector<GeometryObject*>&)
{
  
}

particleIndex 
ParticleCreator::countParticles(const Patch* patch,
				std::vector<GeometryObject*>& d_geom_objs) const
{
  particleIndex sum = 0;
  for(int i=0; i<(int)d_geom_objs.size(); i++)
    sum += countParticles(d_geom_objs[i], patch);
  return sum;
}


particleIndex ParticleCreator::countParticles(GeometryObject* obj,
					      const Patch* patch) const

{
   GeometryPiece* piece = obj->getPiece();
   Box b1 = piece->getBoundingBox();
   Box b2 = patch->getBox();
   Box b = b1.intersect(b2);
   if(b.degenerate())
      return 0;

   IntVector ppc = obj->getNumParticlesPerCell();
   Vector dxpp = patch->dCell()/obj->getNumParticlesPerCell();
   Vector dcorner = dxpp*0.5;
   particleIndex count = 0;

   for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
     Point lower = patch->nodePosition(*iter) + dcorner;
     for(int ix=0;ix < ppc.x(); ix++){
       for(int iy=0;iy < ppc.y(); iy++){
	 for(int iz=0;iz < ppc.z(); iz++){
	   IntVector idx(ix, iy, iz);
	   Point p = lower + dxpp*idx;
	   if(!b2.contains(p))
	     throw InternalError("Particle created outside of patch?");
	   
	   if(piece->inside(p))
	     count++;
	 }
       }
     }
   }
   
   return count;


}

