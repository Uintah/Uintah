//  MPMMaterial.cc

#include "MPMMaterial.h"

#include "ConstitutiveModel.h"
#include "Membrane.h"
#include "ConstitutiveModelFactory.h"

#include <Core/Geometry/IntVector.h>

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
#include <Packages/Uintah/CCA/Components/MPM/Fracture/FractureFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/Fracture/Fracture.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMLabel.h>

#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/ForceBC.h>
#include <Packages/Uintah/CCA/Components/MPM/PhysicalBC/CrackBC.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfState.h>
#include <Packages/Uintah/CCA/Components/ICE/EOS/EquationOfStateFactory.h>
#include <Core/Util/NotFinished.h>

#include <iostream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

MPMMaterial::MPMMaterial(ProblemSpecP& ps, MPMLabel* lb)
  : Material(ps), lb(lb), d_cm(0), d_burn(0), d_fracture(0), d_eos(0)
{
   // Constructor

  // Follow the layout of the input file
  // Steps:
  // 1.  Determine the type of constitutive model and create it.
  // 2.  Get the general properties of the material such as
  //     density, toughness, thermal_conductivity, specific_heat.
  // 3.  Loop through all of the geometry pieces that make up a single
  //     geometry object.
  // 4.  Within the geometry object, assign the boundary conditions
  //     to the object.
  // 5.  Assign the velocity field.

  // Step 1 -- create the constitutive gmodel.

   d_cm = ConstitutiveModelFactory::create(ps,lb);
   if(!d_cm)
      throw ParameterNotFound("No constitutive model");

   Membrane* MEM = dynamic_cast<Membrane*>(d_cm);
   if(MEM){
	d_membrane=true;
   }

   d_burn = BurnFactory::create(ps);
	
   d_fracture = FractureFactory::create(ps);

//   d_eos = EquationOfStateFactory::create(ps);

   // Step 2 -- get the general material properties

   ps->require("density",d_density);
   ps->require("thermal_conductivity",d_thermalConductivity);
   ps->require("specific_heat",d_specificHeat);
   ps->require("specific_heat",d_specificHeat);
   ps->get("gamma",d_gamma);
   
   if(d_fracture) {
     ps->require("pressure_rate",d_pressureRate);
     ps->require("explosive_pressure",d_explosivePressure);
     ps->require("initial_pressure",d_initialPressure);
   }

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
  delete d_fracture;
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

Fracture * MPMMaterial::getFractureModel() const
{
  // Return the pointer to the fracture model associated
  // with this material

  return d_fracture;
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
   new_dw->allocate(position, lb->pXLabel, subset);
   ParticleVariable<Vector> pvelocity;
   new_dw->allocate(pvelocity, lb->pVelocityLabel, subset);
   ParticleVariable<Vector> pexternalforce;
   new_dw->allocate(pexternalforce, lb->pExternalForceLabel, subset);
   ParticleVariable<double> pmass;
   new_dw->allocate(pmass, lb->pMassLabel, subset);
   ParticleVariable<double> pvolume;
   new_dw->allocate(pvolume, lb->pVolumeLabel, subset);
   ParticleVariable<double> ptemperature;
   new_dw->allocate(ptemperature, lb->pTemperatureLabel, subset);
   ParticleVariable<long64> pparticleID;
   new_dw->allocate(pparticleID, lb->pParticleIDLabel, subset);

   ParticleVariable<Vector> pTang1, pTang2, pNorm;
   if(d_membrane){
     new_dw->allocate(pTang1, lb->pTang1Label, subset);
     new_dw->allocate(pTang2, lb->pTang2Label, subset);
     new_dw->allocate(pNorm,  lb->pNormLabel,  subset);
   }

   ParticleVariable<int> pIsBroken;
   ParticleVariable<Vector> pCrackNormal;
   ParticleVariable<Vector> pTipNormal;
   ParticleVariable<Vector> pExtensionDirection;
   ParticleVariable<double> pToughness;
   new_dw->allocate(pToughness, lb->pToughnessLabel, subset);
   ParticleVariable<double> pCrackSurfacePressure;
   ParticleVariable<Vector> pDisplacement;

   if(d_fracture) {
     new_dw->allocate(pIsBroken, lb->pIsBrokenLabel, subset);
     new_dw->allocate(pCrackNormal, lb->pCrackNormalLabel, subset);
     new_dw->allocate(pTipNormal, lb->pTipNormalLabel, subset);
     new_dw->allocate(pExtensionDirection, lb->pExtensionDirectionLabel,subset);
     new_dw->allocate(pCrackSurfacePressure, lb->pCrackSurfacePressureLabel,
                                                             subset);
     new_dw->allocate(pDisplacement, lb->pDisplacementLabel, subset);
   }
   
   particleIndex start = 0;
   if(!d_membrane){
     for(int i=0; i<(int)d_geom_objs.size(); i++){
       start += createParticles( d_geom_objs[i], start, position,
				 pvelocity,pexternalforce,pmass,pvolume,
				 ptemperature,pToughness,
				 pparticleID,cellNAPID,patch);
     }
   }
   else{
     for(int i=0; i<(int)d_geom_objs.size(); i++){
       start += createParticles( d_geom_objs[i], start, position,
				 pvelocity,pexternalforce,pmass,pvolume,
				 ptemperature,pToughness,
				 pparticleID,cellNAPID,patch,
                                 pTang1, pTang2, pNorm);
     }
   }

   particleIndex partclesNum = start;

   for(particleIndex pIdx=0;pIdx<partclesNum;++pIdx) {
     if(d_fracture) {
	pIsBroken[pIdx] = 0;
	pCrackNormal[pIdx] = Vector(0.,0.,0.);
	pTipNormal[pIdx] = Vector(0.,0.,0.);
	pExtensionDirection[pIdx] = Vector(0.,0.,0.);
	pCrackSurfacePressure[pIdx] = 0;
	pDisplacement[pIdx] = Vector(0.,0.,0.);
     }

     //applyPhysicalBCToParticles
     
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

       if(d_fracture) {
       if( !pIsBroken[pIdx] && pTipNormal[pIdx].length2() < 0.5) {
         if (bcs_type == "Crack") {
           CrackBC* bc = dynamic_cast<CrackBC*>
			(MPMPhysicalBCFactory::mpmPhysicalBCs[i]);
	 
	   Vector d = p - bc->origin();
	   double x = Dot(d,bc->e1());
	   double y = Dot(d,bc->e2());
	 
	   Matrix3 mat1(1, bc->x1(), bc->y1(),
	                1, bc->x2(), bc->y2(),
		        1,     x,        y  );
           double mat1v = mat1.Determinant();
	 
  	   Matrix3 mat2(1, bc->x2(), bc->y2(),
	                1, bc->x3(), bc->y3(),
		        1,     x,        y  );
           double mat2v = mat2.Determinant();

	   Matrix3 mat3(1, bc->x3(), bc->y3(),
	                1, bc->x4(), bc->y4(),
		        1,     x,        y  );
           double mat3v = mat3.Determinant();

	   Matrix3 mat4(1, bc->x4(), bc->y4(),
	                1, bc->x1(), bc->y1(),
		        1,     x,        y  );
           double mat4v = mat4.Determinant();

	   bool inside;
           if( mat1v<0 && mat2v<0 && mat3v<0 && mat4v<0 ||
	       mat1v>0 && mat2v>0 && mat3v>0 && mat4v>0 ) inside = true;
           else inside = false;
	 
	   double vdis = Dot( (p - bc->origin()), bc->e3() );

	   double r = pow(pvolume[pIdx], .333)*0.866;

	   if(inside) {
	     if(vdis > 0 && vdis < r) {
	       pIsBroken[pIdx] = 1;
	       pCrackNormal[pIdx] = - bc->e3();
	       pCrackSurfacePressure[pIdx] = d_initialPressure;
	     }
	     else if(vdis <= 0 && vdis >= -r) {
               pIsBroken[pIdx] = 1;
	       pCrackNormal[pIdx] = bc->e3();
	       pCrackSurfacePressure[pIdx] = d_initialPressure;
	     }
	   }
	   else {
	     if( fabs(vdis) < r ) {
	       Point P1(bc->x1(),bc->y1(),0);
	       Point P2(bc->x2(),bc->y2(),0);
  	       Point P3(bc->x3(),bc->y3(),0);
	       Point P4(bc->x4(),bc->y4(),0);
	       Point Q(x,y,0);
	   
	       Vector e12 = P2-P1; e12.normalize();
	       Vector e23 = P3-P2; e23.normalize();
	       Vector e34 = P4-P3; e34.normalize();
	       Vector e41 = P1-P4; e41.normalize();
	   
	       Vector dis;
	   
	       dis = Q-P1;
	       if( ( dis - e12*Dot(dis,e12) ).length() < r*2 && mat4v*mat2v>0 ) 
	       {
	         if(vdis > 0) pTipNormal[pIdx] = -bc->e3();
	         else pTipNormal[pIdx] = bc->e3();
	         pExtensionDirection[pIdx] = bc->e1() * e12[2] - 
		                             bc->e2() * e12[1];
	         if(mat1v>0) pExtensionDirection[pIdx] = 
		   -pExtensionDirection[pIdx];
	       }

	       dis = Q-P2;
	       if( ( dis - e23*Dot(dis,e23) ).length() < r*2  && mat1v*mat3v>0 ) 
	       {
	         if(vdis > 0) pTipNormal[pIdx] = -bc->e3();
	         else pTipNormal[pIdx] = bc->e3();
	         pExtensionDirection[pIdx] = bc->e1() * e23[2] - 
		                             bc->e2() * e23[1];
	         if(mat2v>0) pExtensionDirection[pIdx] = 
		   -pExtensionDirection[pIdx];
	       }

	       dis = Q-P3;
	       if( ( dis - e34*Dot(dis,e34) ).length() < r*2 && mat2v*mat4v>0 )
	       {
	       if(vdis > 0) pTipNormal[pIdx] = -bc->e3();
	       else pTipNormal[pIdx] = bc->e3();
	       pExtensionDirection[pIdx] = bc->e1() * e34[2] - 
	                                   bc->e2() * e34[1];
	       if(mat3v>0) pExtensionDirection[pIdx] = 
	         -pExtensionDirection[pIdx];
	       }

	       dis = Q-P4;
	       if( ( dis - e41*Dot(dis,e41) ).length() < r*2 && mat3v*mat1v>0 )
	       {
	         if(vdis > 0) pTipNormal[pIdx] = -bc->e3();
	         else pTipNormal[pIdx] = bc->e3();
	         pExtensionDirection[pIdx] = bc->e1() * e41[2] - 
		                             bc->e2() * e41[1];
	         if(mat4v>0) pExtensionDirection[pIdx] = 
		   -pExtensionDirection[pIdx];
	       }
	     }
	   }
	 }
       } //"Crack"
       } //fracture
     }
   }

   new_dw->put(position,             lb->pXLabel);
   new_dw->put(pvelocity,            lb->pVelocityLabel);
   new_dw->put(pexternalforce,       lb->pExternalForceLabel);
   new_dw->put(pmass,                lb->pMassLabel);
   new_dw->put(pvolume,              lb->pVolumeLabel);
   new_dw->put(ptemperature,         lb->pTemperatureLabel);
   new_dw->put(pparticleID,          lb->pParticleIDLabel);
   
   if(d_membrane){
     new_dw->put(pTang1, lb->pTang1Label);
     new_dw->put(pTang2, lb->pTang2Label);
     new_dw->put(pNorm,  lb->pNormLabel);
   }

   if(d_fracture) {
     new_dw->put(pIsBroken, lb->pIsBrokenLabel);
     new_dw->put(pCrackNormal, lb->pCrackNormalLabel);
     new_dw->put(pTipNormal, lb->pTipNormalLabel);
     new_dw->put(pExtensionDirection, lb->pExtensionDirectionLabel);
     new_dw->put(pToughness, lb->pToughnessLabel);
     new_dw->put(pCrackSurfacePressure, lb->pCrackSurfacePressureLabel);
     new_dw->put(pDisplacement, lb->pDisplacementLabel);
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
     for(CellIterator iter = patch->getCellIterator(b); !iter.done(); iter++){
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
				   ParticleVariable<double>& toughness,
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

   particleIndex count = 0;
     for(CellIterator iter = patch->getCellIterator(b); !iter.done(); iter++){
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
		  short int& myCellNAPID = cellNAPID[cell_idx];
		  particleID[start+count] = cellID | (long64)myCellNAPID;
		  ASSERT(myCellNAPID < 0x7fff);
		  myCellNAPID++;

		  if( d_fracture ) {
		    if(obj->getToughnessMin() == obj->getToughnessMax())
		    {
		      toughness[start+count] = obj->getToughnessMin();
		    }
		    else
		    {
	              double probability;
		      double x;
		      double toughnessAve = ( obj->getToughnessMin() + 
                                 obj->getToughnessMax() )/2;
		      double toughnessWid = ( obj->getToughnessMax() - 
                                 obj->getToughnessMin() )/2 *
                                 obj->getToughnessVariation();
		      double s;
		      do {
	                double rand = drand48();
	                s = (1-rand) * obj->getToughnessMin() + 
		            rand * obj->getToughnessMax();
	              
	                probability = drand48();
	                x = (s-toughnessAve)/toughnessWid;
	              } while( exp(-x*x) < probability );
	              toughness[start+count] = s;
		    }
		  }

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
				   ParticleVariable<double>& toughness,
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

   particleIndex count = 0;
   SphereMembraneGeometryPiece* SMGP =
                              dynamic_cast<SphereMembraneGeometryPiece*>(piece);
   if(SMGP){
        int numP = SMGP->createParticles(patch, position, volume,
                                         ptang1, ptang2, pnorm, start);
        for(int idx=0;idx<(start+numP);idx++){
            velocity[start+idx]=obj->getInitialVelocity();
            temperature[start+idx]=obj->getInitialTemperature();
            mass[start+idx]=d_density * volume[start+idx];
            // Determine if particle is on the surface
            pexternalforce[start+idx]=Vector(0,0,0); // for now
            toughness[start+count] = obj->getToughnessMin();
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
               cerr << "cellID is fucked up" << endl;
            }
        }
   }
   else{
     for(CellIterator iter = patch->getCellIterator(b); !iter.done(); iter++){
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
                  toughness[start+count] = obj->getToughnessMin();
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
int MPMMaterial::checkForSurface(const GeometryPiece* piece, const Point p,
							const Vector dxpp)
{

//  Check the candidate points which surround the point just passed
//  in.  If any of those points are not also inside the object
//  the current point is on the surface

  int ss = 0;

  // Check to the left (-x)
  if(!piece->inside(p-Vector(dxpp.x(),0.,0.)))
    ss++;
  // Check to the right (+x)
  if(!piece->inside(p+Vector(dxpp.x(),0.,0.)))
    ss++;
  // Check behind (-y)
  if(!piece->inside(p-Vector(0.,dxpp.y(),0.)))
    ss++;
  // Check in front (+y)
  if(!piece->inside(p+Vector(0.,dxpp.y(),0.)))
    ss++;
  // Check below (-z)
  if(!piece->inside(p-Vector(0.,0.,dxpp.z())))
    ss++;
  // Check above (+z)
  if(!piece->inside(p+Vector(0.,0.,dxpp.z())))
    ss++;

  if(ss>0){
    return 1;
  }
  else {
    return 0;
  }
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

double MPMMaterial::getPressureRate() const
{
  return d_pressureRate;
}

double MPMMaterial::getExplosivePressure() const
{
  return d_explosivePressure;
}

double MPMMaterial::getInitialPressure() const
{
  return d_initialPressure;
}
