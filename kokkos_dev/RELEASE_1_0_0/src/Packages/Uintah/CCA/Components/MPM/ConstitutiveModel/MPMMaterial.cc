//  MPMMaterial.cc

#include "MPMMaterial.h"

#include "ConstitutiveModel.h"
#include "ConstitutiveModelFactory.h"

#include <Core/Geometry/IntVector.h>

#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Packages/Uintah/Core/Grid/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/UnionGeometryPiece.h>
#include <Packages/Uintah/CCA/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Packages/Uintah/Core/Exceptions/ParameterNotFound.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/CCA/Components/MPM/Util/Matrix3.h>
#include <Packages/Uintah/CCA/Components/MPMICE/Combustion/BurnFactory.h>
#include <Packages/Uintah/CCA/Components/MPMICE/Combustion/Burn.h>
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
  : lb(lb)
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

   d_cm = ConstitutiveModelFactory::create(ps);
   if(!d_cm)
      throw ParameterNotFound("No constitutive model");

#if 0
   d_burn = BurnFactory::create(ps);
#else
   d_burn = 0;
   NOT_FINISHED("Burn busted - Steve");
#endif
	
   d_fracture = FractureFactory::create(ps);

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
  delete d_fracture;

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
				  PerPatch<long> NAPID,
				  const Patch* patch,
				  DataWarehouse* new_dw)
{
  //   const MPMLabel* lb = MPMLabel::getLabels();

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
   ParticleVariable<int> pissurf;
//   new_dw->allocate(pissurf, lb->pSurfLabel, subset);
   ParticleVariable<double> ptemperature;
   new_dw->allocate(ptemperature, lb->pTemperatureLabel, subset);
   ParticleVariable<long> pparticleID;
   new_dw->allocate(pparticleID, lb->pParticleIDLabel, subset);

   ParticleVariable<int> pIsBroken;
   ParticleVariable<Vector> pCrackNormal1;
   ParticleVariable<Vector> pCrackNormal2;
   ParticleVariable<Vector> pCrackNormal3;
   ParticleVariable<double> pToughness;
   
   if(d_fracture) {
     new_dw->allocate(pIsBroken, 
       lb->pIsBrokenLabel, subset);
     new_dw->allocate(pCrackNormal1, 
       lb->pCrackNormal1Label, subset);
     new_dw->allocate(pCrackNormal2, 
       lb->pCrackNormal2Label, subset);
     new_dw->allocate(pCrackNormal3, 
       lb->pCrackNormal3Label, subset);
     new_dw->allocate(pToughness, 
       lb->pToughnessLabel, subset);
   }
   
   particleIndex start = 0;
   for(int i=0; i<(int)d_geom_objs.size(); i++){
      start += createParticles( d_geom_objs[i], start, position,
				pvelocity,pexternalforce,pmass,pvolume,
				pissurf,ptemperature,pToughness,
				pparticleID,NAPID,patch);
   }

   particleIndex partclesNum = start;

   ParticleVariable<Vector> ptemperatureGradient;
   new_dw->allocate(ptemperatureGradient, lb->pTemperatureGradientLabel, subset);

   ParticleVariable<double> pexternalHeatRate;
   new_dw->allocate(pexternalHeatRate, lb->pExternalHeatRateLabel, subset);

   for(particleIndex pIdx=0;pIdx<partclesNum;++pIdx) {
     ptemperatureGradient[pIdx] = Vector(0.,0.,0.);
     pexternalHeatRate[pIdx] = 0.;
     
     if(d_fracture) {
	pIsBroken[pIdx] = 0;
	pCrackNormal1[pIdx] = Vector(0.,0.,0.);
	pCrackNormal2[pIdx] = Vector(0.,0.,0.);
	pCrackNormal3[pIdx] = Vector(0.,0.,0.);
     }

     pexternalforce[pIdx] = Vector(0.0,0.0,0.0);

     //applyPhysicalBCToParticles
     
     const Point& p( position[pIdx] );
     
     for (int i = 0; i<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); i++ ) {
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
               //cout << pexternalforce[pIdx] << endl;
         }
       }

       if(d_fracture) {
       if (bcs_type == "Crack") {
         CrackBC* bc = dynamic_cast<CrackBC*>
			(MPMPhysicalBCFactory::mpmPhysicalBCs[i]);
	 
	 Vector d = p - bc->origin();
	 double x = Dot(d,bc->e1());
	 double y = Dot(d,bc->e2());
	 
	 {
	   Matrix3 mat(1, bc->x1(), bc->y1(),
	               1, bc->x2(), bc->y2(),
		       1,     x,        y  );
           if( mat.Determinant() < 0 ) continue;
	 }

	 {
	   Matrix3 mat(1, bc->x2(), bc->y2(),
	               1, bc->x3(), bc->y3(),
		       1,     x,        y  );
           if( mat.Determinant() < 0 ) continue;
	 }
	 
	 {
	   Matrix3 mat(1, bc->x3(), bc->y3(),
	               1, bc->x4(), bc->y4(),
		       1,     x,        y  );
           if( mat.Determinant() < 0 ) continue;
	 }

	 {
	   Matrix3 mat(1, bc->x4(), bc->y4(),
	               1, bc->x1(), bc->y1(),
		       1,     x,        y  );
           if( mat.Determinant() < 0 ) continue;
	 }
	 
	 double vdis = Dot( (p - bc->origin()), bc->e3() );

	 double particle_half_size = pow( pvolume[pIdx], 1./3.) /2;

	 if(vdis > 0 && vdis < particle_half_size * 2) {
	   pIsBroken[pIdx]++;
	        if(pIsBroken[pIdx]==1) pCrackNormal1[pIdx] = - bc->e3();
	   else if(pIsBroken[pIdx]==2) pCrackNormal2[pIdx] = - bc->e3();
	   else if(pIsBroken[pIdx]==3) pCrackNormal3[pIdx] = - bc->e3();
	 }
	 else if(vdis <= 0 && vdis >= -particle_half_size*2) {
	   pIsBroken[pIdx]++;
	        if(pIsBroken[pIdx]==1) pCrackNormal1[pIdx] = bc->e3();
	   else if(pIsBroken[pIdx]==2) pCrackNormal2[pIdx] = bc->e3();
	   else if(pIsBroken[pIdx]==3) pCrackNormal3[pIdx] = bc->e3();
	 }
       } //"Crack"
       } //fracture
     }
   }

   new_dw->put(ptemperatureGradient, lb->pTemperatureGradientLabel);
   new_dw->put(pexternalHeatRate, lb->pExternalHeatRateLabel);

   new_dw->put(position, lb->pXLabel);
   new_dw->put(pvelocity, lb->pVelocityLabel);
   new_dw->put(pexternalforce, lb->pExternalForceLabel);
   new_dw->put(pmass, lb->pMassLabel);
   new_dw->put(pvolume, lb->pVolumeLabel);
//   new_dw->put(pissurf, lb->pSurfLabel);
   new_dw->put(ptemperature, lb->pTemperatureLabel);
   new_dw->put(pparticleID, lb->pParticleIDLabel);
   
   if(d_fracture) {
     new_dw->put(pIsBroken, lb->pIsBrokenLabel);
     new_dw->put(pCrackNormal1, lb->pCrackNormal1Label);
     new_dw->put(pCrackNormal2, lb->pCrackNormal2Label);
     new_dw->put(pCrackNormal3, lb->pCrackNormal3Label);
     new_dw->put(pToughness, lb->pToughnessLabel);
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
   cerr << "Creating " << count << " particles on patch: " << patch->getID() << '\n';
   return count;
}


particleIndex MPMMaterial::createParticles(GeometryObject* obj,
				   particleIndex start,
				   ParticleVariable<Point>& position,
				   ParticleVariable<Vector>& velocity,
				   ParticleVariable<Vector>& pexternalforce,
				   ParticleVariable<double>& mass,
				   ParticleVariable<double>& volume,
				   ParticleVariable<int>& pissurf,
				   ParticleVariable<double>& temperature,
				   ParticleVariable<double>& toughness,
				   ParticleVariable<long>& particleID,
				   PerPatch<long>& NAPID,
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
   int nbits=40;

   long patch_number = patch->getID();

   patch_number <<= nbits;

   particleIndex count = 0;
   for(CellIterator iter = patch->getCellIterator(b); !iter.done(); iter++){
      Point lower = patch->nodePosition(*iter) + dcorner;
      for(int ix=0;ix < ppc.x(); ix++){
	 for(int iy=0;iy < ppc.y(); iy++){
	    for(int iz=0;iz < ppc.z(); iz++){
	       IntVector idx(ix, iy, iz);
	       Point p = lower + dxpp*idx;
	       if(piece->inside(p)){
		  position[start+count]=p;
		  volume[start+count]=dxpp.x()*dxpp.y()*dxpp.z();
		  velocity[start+count]=obj->getInitialVelocity();
		  temperature[start+count]=obj->getInitialTemperature();
		  mass[start+count]=d_density * volume[start+count];
		  // Determine if particle is on the surface
//		  pissurf[start+count]=checkForSurface(piece,p,dxpp);
		  pexternalforce[start+count]=Vector(0,0,0); // for now
		  particleID[start+count]=
				(patch_number | (NAPID + start + count));


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
	       }
	    }
	 }
      }
   }
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

