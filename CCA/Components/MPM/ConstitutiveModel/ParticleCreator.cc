#include "ParticleCreator.h"
#include <Packages/Uintah/CCA/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/CrackBC.h>

using namespace Uintah;
using std::vector;

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
				      vector<GeometryObject*>&)
{
  
}

particleIndex 
ParticleCreator::countParticles(const Patch* patch,
				vector<GeometryObject*>& d_geom_objs) const
{
  particleIndex sum = 0;
  vector<GeometryObject*>::const_iterator geom;
  for (geom=d_geom_objs.begin(); geom != d_geom_objs.end(); ++geom) 
    sum += countParticles(*geom,patch);

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

void ParticleCreator::applyForceBC(particleIndex particlesNum, 
				   ParticleVariable<Vector>& pextforce,
				   ParticleVariable<double>& pmass,
				   ParticleVariable<Point>& position)
{

  for(particleIndex pIdx=0;pIdx<particlesNum;++pIdx) {
     pextforce[pIdx] = Vector(0.0,0.0,0.0);

     const Point& p( position[pIdx] );
     
     for (int i = 0; i<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); i++){
       string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[i]->getType();
        
       if (bcs_type == "Force") {
         ForceBC* bc = dynamic_cast<ForceBC*>
			(MPMPhysicalBCFactory::mpmPhysicalBCs[i]);

         const Point& lower( bc->getLowerRange() );
         const Point& upper( bc->getUpperRange() );
          
         if(lower.x()<= p.x() && p.x() <= upper.x() &&
            lower.y()<= p.y() && p.y() <= upper.y() &&
            lower.z()<= p.z() && p.z() <= upper.z() ){
               pextforce[pIdx] = bc->getForceDensity() * pmass[pIdx];
         }
       }
     }
  }


}

