//  MPMMaterial.cc

#include "MPMMaterial.h"

#include "ConstitutiveModel.h"
#include "Membrane.h"
#include "ConstitutiveModelFactory.h"

#include <Core/Geometry/IntVector.h>

#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/UnionGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/SphereMembraneGeometryPiece.h>
#include <Packages/Uintah/CCA/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/CCA/Components/HETransformation/BurnFactory.h>
#include <Packages/Uintah/CCA/Components/HETransformation/Burn.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>

#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/CrackBC.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfState.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfStateFactory.h>
#if 0
#include "ImplicitParticleCreator.h"
#include "DefaultParticleCreator.h"
#include "MembraneParticleCreator.h"
#endif
#include <Core/Util/NotFinished.h>
#include <iostream>

#define d_TINY_RHO 1.0e-100 // also defined  ICE.cc and ICEMaterial.cc 

using namespace std;
using namespace Uintah;
using namespace SCIRun;

MPMMaterial::MPMMaterial(ProblemSpecP& ps, MPMLabel* lb, int n8or27,
			 string integrator)
  : Material(ps), lb(lb), d_cm(0), d_burn(0), d_eos(0), d_membrane(false)
#if 0
    d_particle_creator(0)
#endif
{
   // Constructor

  // Follow the layout of the input file
  // Steps:
  // 1.  Determine the type of constitutive model and create it.
  // 2.  Get the general properties of the material such as
  //     density, thermal_conductivity, specific_heat.
  // 3.  Loop through all of the geometry pieces that make up a single
  //     geometry object.
  // 4.  Within the geometry object, assign the boundary conditions
  //     to the object.
  // 5.  Assign the velocity field.

  // Step 1 -- create the constitutive gmodel.

   d_cm = ConstitutiveModelFactory::create(ps,lb,n8or27,integrator);
   if(!d_cm)
      throw ParameterNotFound("No constitutive model");

   Membrane* MEM = dynamic_cast<Membrane*>(d_cm);
   if(MEM){
	d_membrane=true;
   }

   d_burn = BurnFactory::create(ps);
#if 0
   // Check to see which ParticleCreator object we need
   if (integrator == "implicit") 
     d_particle_creator = scinew ImplicitParticleCreator();
   else if (dynamic_cast<Membrane*>(d_cm) != 0)
     d_particle_creator = scinew MembraneParticleCreator();
   else
     d_particle_creator = scinew DefaultParticleCreator();
#endif
	
//   d_eos = EquationOfStateFactory::create(ps);

   // Step 2 -- get the general material properties

   ps->require("density",d_density);
   ps->require("thermal_conductivity",d_thermalConductivity);
   ps->require("specific_heat",d_specificHeat);
   ps->require("specific_heat",d_specificHeat);
   ps->get("gamma",d_gamma);
   
   // Step 3 -- Loop through all of the pieces in this geometry object
   int piece_num = 0;
   for (ProblemSpecP geom_obj_ps = ps->findBlock("geom_object");
	geom_obj_ps != 0; 
	geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

      vector<GeometryPiece*> pieces;
      GeometryPieceFactory::create(geom_obj_ps, pieces);

      GeometryPiece* mainpiece;
      if(pieces.size() == 0){
	 throw ParameterNotFound("No piece specified in geom_object");
      } else if(pieces.size() > 1){
	 mainpiece = scinew UnionGeometryPiece(pieces);
      } else {
	 mainpiece = pieces[0];
      }

      piece_num++;
      d_geom_objs.push_back(scinew GeometryObject(this,mainpiece, geom_obj_ps));
   }
}

MPMMaterial::~MPMMaterial()
{
  // Destructor

  delete d_cm;
  delete d_burn;
  delete d_eos;

  for (int i = 0; i<(int)d_geom_objs.size(); i++) {
    delete d_geom_objs[i];
  }
}

ConstitutiveModel * MPMMaterial::getConstitutiveModel() const
{
  // Return the pointer to the constitutive model associated
  // with this material

  return d_cm;
}

Burn * MPMMaterial::getBurnModel()
{
  // Return the pointer to the burn model associated
  // with this material

  return d_burn;
}

EquationOfState* MPMMaterial::getEOSModel() const
{
  // Return the pointer to the EOS model associated
  // with this material

  return d_eos;
}

particleIndex MPMMaterial::countParticles(const Patch* patch) const
{
   particleIndex sum = 0;
   for(int i=0; i<(int)d_geom_objs.size(); i++)
      sum+= countParticles(d_geom_objs[i], patch);
   return sum;
}

void MPMMaterial::createParticles(particleIndex numParticles,
				  CCVariable<short int>& cellNAPID,
				  const Patch* patch,
				  DataWarehouse* new_dw)
{
   ParticleSubset* subset = new_dw->createParticleSubset(numParticles,
							 getDWIndex(), patch);
   ParticleVariable<Point> position;
   new_dw->allocateAndPut(position, lb->pXLabel, subset);
   ParticleVariable<Vector> pvelocity;
   new_dw->allocateAndPut(pvelocity, lb->pVelocityLabel, subset);
#ifdef IMPLICIT
   ParticleVariable<Vector> pacceleration;
   new_dw->allocateAndPut(pacceleration, lb->pAccelerationLabel, subset);
   ParticleVariable<double> pvolumeold;
   new_dw->allocateAndPut(pvolumeold, lb->pVolumeOldLabel, subset);
   ParticleVariable<Matrix3> bElBar;
   new_dw->allocateTemporary(bElBar,  subset);
#endif
   ParticleVariable<Vector> pexternalforce;
   new_dw->allocateAndPut(pexternalforce, lb->pExternalForceLabel, subset);
   ParticleVariable<double> pmass;
   new_dw->allocateAndPut(pmass, lb->pMassLabel, subset);
   ParticleVariable<double> pvolume;
   new_dw->allocateAndPut(pvolume, lb->pVolumeLabel, subset);
   ParticleVariable<double> ptemperature;
   new_dw->allocateAndPut(ptemperature, lb->pTemperatureLabel, subset);
   ParticleVariable<long64> pparticleID;
   new_dw->allocateAndPut(pparticleID, lb->pParticleIDLabel, subset);
   ParticleVariable<Vector> psize;
   new_dw->allocateAndPut(psize, lb->pSizeLabel, subset);

   ParticleVariable<Vector> pTang1, pTang2, pNorm;
   if(d_membrane){
     new_dw->allocateAndPut(pTang1, lb->pTang1Label, subset);
     new_dw->allocateAndPut(pTang2, lb->pTang2Label, subset);
     new_dw->allocateAndPut(pNorm, lb->pNormLabel,  subset);
   }

   particleIndex start = 0;
   if(!d_membrane){
     for(int i=0; i<(int)d_geom_objs.size(); i++){
       start += createParticles( d_geom_objs[i], start, position,
				 pvelocity,pexternalforce,pmass,pvolume,
				 ptemperature,psize,
#ifdef IMPLICIT
				 pacceleration,
				 pvolumeold,
				 bElBar,
#endif
				 pparticleID,cellNAPID,patch);
     }
   }
   else{
     for(int i=0; i<(int)d_geom_objs.size(); i++){
       start += createParticles( d_geom_objs[i], start, position,
				 pvelocity,pexternalforce,pmass,pvolume,
				 ptemperature,psize,
				 pparticleID,cellNAPID,patch,
                                 pTang1, pTang2, pNorm);
     }
   }

   particleIndex partclesNum = start;

   for(particleIndex pIdx=0;pIdx<partclesNum;++pIdx) {
     pexternalforce[pIdx] = Vector(0.0,0.0,0.0);

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
               pexternalforce[pIdx] = bc->getForceDensity() * pmass[pIdx];
         }
       }
     }
   }

}

particleIndex MPMMaterial::countParticles(GeometryObject* obj,
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

   SphereMembraneGeometryPiece* SMGP =
                              dynamic_cast<SphereMembraneGeometryPiece*>(piece);
   if(SMGP){
        count = SMGP->returnParticleCount(patch);
   }
   else{
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
   }
   return count;
}

particleIndex MPMMaterial::createParticles(GeometryObject* obj,
				   particleIndex start,
				   ParticleVariable<Point>& position,
				   ParticleVariable<Vector>& velocity,
				   ParticleVariable<Vector>& pexternalforce,
				   ParticleVariable<double>& mass,
				   ParticleVariable<double>& volume,
				   ParticleVariable<double>& temperature,
				   ParticleVariable<Vector>& psize,
#ifdef IMPLICIT
				   ParticleVariable<Vector>& pacceleration,
				   ParticleVariable<double>& pvolumeold,
				   ParticleVariable<Matrix3>& bElBar,
#endif
				   ParticleVariable<long64>& particleID,
				   CCVariable<short int>& cellNAPID,
				   const Patch* patch)
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
   // Size as a fraction of the cell size
   Vector size(1./((double) ppc.x()),
               1./((double) ppc.y()),
               1./((double) ppc.z()));

   particleIndex count = 0;
     for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      Point lower = patch->nodePosition(*iter) + dcorner;
      for(int ix=0;ix < ppc.x(); ix++){
	 for(int iy=0;iy < ppc.y(); iy++){
	    for(int iz=0;iz < ppc.z(); iz++){
	       IntVector idx(ix, iy, iz);
	       Point p = lower + dxpp*idx;
	       IntVector cell_idx = iter.index();
	       // If the assertion fails then we may just need to change
	       // the format of particle ids such that the cell indices
	       // have more bits.
	       ASSERT(cell_idx.x() <= 0xffff && cell_idx.y() <= 0xffff
		      && cell_idx.z() <= 0xffff);
	       long64 cellID = ((long64)cell_idx.x() << 16) |
                               ((long64)cell_idx.y() << 32) |
                               ((long64)cell_idx.z() << 48);
	       if(piece->inside(p)){
		  position[start+count]=p;
		  volume[start+count]=dxpp.x()*dxpp.y()*dxpp.z();
		  velocity[start+count]=obj->getInitialVelocity();
#ifdef IMPLICIT
		  pacceleration[start+count]=Vector(0.,0.,0.);
		  pvolumeold[start+count]=volume[start+count];
		  bElBar[start+count]=Matrix3(0.);
#endif
		  temperature[start+count]=obj->getInitialTemperature();
		  mass[start+count]=d_density * volume[start+count];
		  // Determine if particle is on the surface
		  pexternalforce[start+count]=Vector(0,0,0); // for now
		  short int& myCellNAPID = cellNAPID[cell_idx];
		  particleID[start+count] = cellID | (long64)myCellNAPID;
                  psize[start+count] = size;
		  ASSERT(myCellNAPID < 0x7fff);
		  myCellNAPID++;
		  count++;
	       }  // if inside
	    }  // loop in z
	 }  // loop in y
      }  // loop in x
     } // for
   return count;
}

particleIndex MPMMaterial::createParticles(GeometryObject* obj,
				   particleIndex start,
				   ParticleVariable<Point>& position,
				   ParticleVariable<Vector>& velocity,
				   ParticleVariable<Vector>& pexternalforce,
				   ParticleVariable<double>& mass,
				   ParticleVariable<double>& volume,
				   ParticleVariable<double>& temperature,
				   ParticleVariable<Vector>& psize,
				   ParticleVariable<long64>& particleID,
				   CCVariable<short int>& cellNAPID,
				   const Patch* patch,
				   ParticleVariable<Vector>& ptang1,
				   ParticleVariable<Vector>& ptang2,
				   ParticleVariable<Vector>& pnorm)
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
   // Size as a fraction of the cell size
   Vector size(1./((double) ppc.x()),
               1./((double) ppc.y()),
               1./((double) ppc.z()));

   particleIndex count = 0;
   SphereMembraneGeometryPiece* SMGP =
                              dynamic_cast<SphereMembraneGeometryPiece*>(piece);
   if(SMGP){
        int numP = SMGP->createParticles(patch, position, volume,
                                         ptang1, ptang2, pnorm, psize, start);
        for(int idx=0;idx<(start+numP);idx++){
            velocity[start+idx]=obj->getInitialVelocity();
            temperature[start+idx]=obj->getInitialTemperature();
            mass[start+idx]=d_density * volume[start+idx];
            // Determine if particle is on the surface
            pexternalforce[start+idx]=Vector(0,0,0); // for now
	    IntVector cell_idx;
            if(patch->findCell(position[start+idx],cell_idx)){
               long64 cellID = ((long64)cell_idx.x() << 16) |
                               ((long64)cell_idx.y() << 32) |
                               ((long64)cell_idx.z() << 48);
               short int& myCellNAPID = cellNAPID[cell_idx];
               ASSERT(myCellNAPID < 0x7fff);
               myCellNAPID++;
               particleID[start+idx] = cellID | (long64)myCellNAPID;
            }
            else{
               cerr << "cellID is not right" << endl;
            }
        }
   }
   else{
     for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++){
      Point lower = patch->nodePosition(*iter) + dcorner;
      for(int ix=0;ix < ppc.x(); ix++){
	 for(int iy=0;iy < ppc.y(); iy++){
	    for(int iz=0;iz < ppc.z(); iz++){
	       IntVector idx(ix, iy, iz);
	       Point p = lower + dxpp*idx;
	       IntVector cell_idx = iter.index();
	       // If the assertion fails then we may just need to change
	       // the format of particle ids such that the cell indices
	       // have more bits.
	       ASSERT(cell_idx.x() <= 0xffff && cell_idx.y() <= 0xffff
		      && cell_idx.z() <= 0xffff);
	       long64 cellID = ((long64)cell_idx.x() << 16) |
                               ((long64)cell_idx.y() << 32) |
                               ((long64)cell_idx.z() << 48);
	       if(piece->inside(p)){
		  position[start+count]=p;
		  volume[start+count]=dxpp.x()*dxpp.y()*dxpp.z();
		  velocity[start+count]=obj->getInitialVelocity();
		  temperature[start+count]=obj->getInitialTemperature();
		  mass[start+count]=d_density * volume[start+count];
		  // Determine if particle is on the surface
		  pexternalforce[start+count]=Vector(0,0,0); // for now
                  psize[start+count] = size;
                  ptang1[start+count] = Vector(1,0,0);
                  ptang2[start+count] = Vector(0,0,1);
                  pnorm[start+count]  = Vector(0,1,0);
		  short int& myCellNAPID = cellNAPID[cell_idx];
		  particleID[start+count] = cellID | (long64)myCellNAPID;
		  ASSERT(myCellNAPID < 0x7fff);
		  myCellNAPID++;

		  count++;

	       }  // if inside
	    }  // loop in z
	 }  // loop in y
      }  // loop in x
     } // for
   } // else
   return count;
}

double MPMMaterial::getThermalConductivity() const
{
  return d_thermalConductivity;
}

double MPMMaterial::getSpecificHeat() const
{
  return d_specificHeat;
}

double MPMMaterial::getHeatTransferCoefficient() const
{
  return d_heatTransferCoefficient;
}

double MPMMaterial::getGamma() const
{
  return d_gamma;
}

double MPMMaterial::getInitialDensity() const
{
  return d_density;
}
/* --------------------------------------------------------------------- 
 Function~  MPMMaterial::initializeCells--
 Notes:  This function initializeCCVariables.  Reasonable values for 
 CC Variables need to be present in all the cells and evolve, even though
 there is no mass.  This is essentially the same routine that is in
 ICEMaterial.cc
_____________________________________________________________________*/
void MPMMaterial::initializeCCVariables(CCVariable<double>& rho_micro,
                                  CCVariable<double>& rho_CC,
                                  CCVariable<double>& temp,
                                  CCVariable<Vector>& vel_CC,
                                  int numMatls,
                                  const Patch* patch)
{ 
  // initialize to -9 so bullet proofing will catch it any cell that
  // isn't initialized
  vel_CC.initialize(Vector(0.,0.,0.));
  rho_micro.initialize(-9.0);
  rho_CC.initialize(-9.0);
  temp.initialize(-9.0);
  Vector dx = patch->dCell();
  
  for(int obj=0; obj<(int)d_geom_objs.size(); obj++){
   GeometryPiece* piece = d_geom_objs[obj]->getPiece();
   Box b1 = piece->getBoundingBox();
   Box b2 = patch->getBox();
   Box b = b1.intersect(b2);
   // Find the bounds of a region a little bigger than the piece's BBox.
   Point b1low(b1.lower().x()-3.*dx.x(),b1.lower().y()-3.*dx.y(),
                                        b1.lower().z()-3.*dx.z());
   Point b1up(b1.upper().x()+3.*dx.x(),b1.upper().y()+3.*dx.y(),
                                        b1.upper().z()+3.*dx.z());
   
   if(b.degenerate()){
      cerr << "b.degenerate" << endl;
      cerr << "So what? " << endl;
   }

   IntVector ppc = d_geom_objs[obj]->getNumParticlesPerCell();
   Vector dxpp    = patch->dCell()/ppc;
   Vector dcorner = dxpp*0.5;
   double totalppc = ppc.x()*ppc.y()*ppc.z();

  for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
     Point lower = patch->nodePosition(*iter) + dcorner;
     int count = 0;
     for(int ix=0;ix < ppc.x(); ix++){
       for(int iy=0;iy < ppc.y(); iy++){
        for(int iz=0;iz < ppc.z(); iz++){
          IntVector idx(ix, iy, iz);
          Point p = lower + dxpp*idx;
          if(piece->inside(p))
            count++;
        }
       }
     }
  //__________________________________
  // For single materials with more than one object 
      if(numMatls == 1)  {
        if ( count > 0  && obj == 0) {
         // vol_frac_CC[*iter]= 1.0;
          vel_CC[*iter]     = d_geom_objs[obj]->getInitialVelocity();
          rho_micro[*iter]  = getInitialDensity();
          rho_CC[*iter]     = rho_micro[*iter] + d_TINY_RHO;
          temp[*iter]       = d_geom_objs[obj]->getInitialTemperature();
        }

        if (count > 0 && obj > 0) {
         // vol_frac_CC[*iter]= 1.0;
          vel_CC[*iter]     = d_geom_objs[obj]->getInitialVelocity();
          rho_micro[*iter]  = getInitialDensity();
          rho_CC[*iter]     = rho_micro[*iter] + d_TINY_RHO;
          temp[*iter]       = d_geom_objs[obj]->getInitialTemperature();
        } 
      }   
      if (numMatls > 1 ) {
        double vol_frac_CC= count/totalppc;       
        rho_micro[*iter]  = getInitialDensity();
        rho_CC[*iter]     = rho_micro[*iter] * vol_frac_CC + d_TINY_RHO;
        temp[*iter]       = 300.0;         
        Point pd = patch->cellPosition(*iter);
        if((pd.x() > b1low.x() && pd.y() > b1low.y() && pd.z() > b1low.z()) &&
           (pd.x() < b1up.x()  && pd.y() < b1up.y()  && pd.z() < b1up.z())){
            vel_CC[*iter]     = d_geom_objs[obj]->getInitialVelocity();
            temp[*iter]       = d_geom_objs[obj]->getInitialTemperature();
        }    
      }    
    }  // Loop over domain
  }  // Loop over geom_objects
}
