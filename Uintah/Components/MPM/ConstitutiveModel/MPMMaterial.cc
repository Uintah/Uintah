//  MPMMaterial.cc
//

#include "MPMMaterial.h"
#include "ConstitutiveModel.h"
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/CellIterator.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/PerPatch.h>
#include <Uintah/Grid/GeometryPieceFactory.h>
#include <Uintah/Grid/UnionGeometryPiece.h>
#include <Uintah/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Uintah/Exceptions/ParameterNotFound.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <iostream>
#include "ConstitutiveModelFactory.h"
#include <Uintah/Components/MPM/Burn/HEBurnFactory.h>
#include <Uintah/Components/MPM/Burn/HEBurn.h>
#include <Uintah/Components/MPM/Fracture/FractureFactory.h>
#include <Uintah/Components/MPM/Fracture/Fracture.h>
#include <Uintah/Components/MPM/MPMLabel.h>

#include <Uintah/Components/MPM/PhysicalBC/MPMPhysicalBCFactory.h>
#include <Uintah/Components/MPM/PhysicalBC/ForceBC.h>

using namespace std;
using namespace Uintah::MPM;
using namespace Uintah;
using namespace SCICore::Geometry;

MPMMaterial::MPMMaterial(ProblemSpecP& ps)
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

   d_burn = HEBurnFactory::create(ps);
   if (!d_burn)
	throw ParameterNotFound("No burn model");
	
   d_fracture = FractureFactory::create(ps);

   // Step 2 -- get the general material properties

   ps->require("density",d_density);
   ps->require("thermal_conductivity",d_thermalConductivity);
   ps->require("specific_heat",d_specificHeat);

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
      
      // Step 4 -- Assign the boundary conditions to the object

      
      // Step 5 -- Assign the velocity field
      int vf;
      ps->require("velocity_field",vf);
      setVFIndex(vf);
   }

   lb = scinew MPMLabel();
}

MPMMaterial::~MPMMaterial()
{
  // Destructor

  delete d_cm;
  delete lb;
  delete d_burn;

  for (int i = 0; i<(int)d_geom_objs.size(); i++) {
    GeometryObject* obj = d_geom_objs[i];
    delete obj;
    delete d_geom_objs[i];
  }
}

ConstitutiveModel * MPMMaterial::getConstitutiveModel() const
{
  // Return the pointer to the constitutive model associated
  // with this material

  return d_cm;
}

HEBurn * MPMMaterial::getBurnModel() const
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
				  DataWarehouseP& new_dw)
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
   ParticleVariable<Vector> pCrackSurfaceNormal;
   ParticleVariable<Vector> pCrackSurfaceContactForce;
   ParticleVariable<double> pTensileStrength;
   
   if(d_fracture) {
     new_dw->allocate(pIsBroken, lb->pIsBrokenLabel, subset);
     new_dw->allocate(pCrackSurfaceNormal, lb->pCrackSurfaceNormalLabel, subset);
     new_dw->allocate(pCrackSurfaceContactForce, lb->pCrackSurfaceContactForceLabel, subset);
     new_dw->allocate(pTensileStrength, lb->pTensileStrengthLabel, subset);
   }
   
   particleIndex start = 0;
   for(int i=0; i<(int)d_geom_objs.size(); i++){
      start += createParticles( d_geom_objs[i], start, position,
				pvelocity,pexternalforce,pmass,pvolume,
				pissurf,ptemperature,pTensileStrength,
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
	pCrackSurfaceNormal[pIdx] = Vector(0.,0.,0.);
	pCrackSurfaceContactForce[pIdx] = Vector(0.,0.,0.);
     }

     pexternalforce[pIdx] = Vector(0.0,0.0,0.0);

     //applyPhysicalBCToParticles
     for (int i = 0; i<(int)MPMPhysicalBCFactory::mpmPhysicalBCs.size(); i++ ) {
       string bcs_type = MPMPhysicalBCFactory::mpmPhysicalBCs[i]->getType();
        
       if (bcs_type == "Force") {
         ForceBC* bc = dynamic_cast<ForceBC*>
			(MPMPhysicalBCFactory::mpmPhysicalBCs[i]);

         const Point& lower( bc->getLowerRange() );
         const Point& upper( bc->getUpperRange() );
          
         if(lower.x()<= position[pIdx].x() && position[pIdx].x() <= upper.x() &&
            lower.y()<= position[pIdx].y() && position[pIdx].y() <= upper.y() &&
            lower.z()<= position[pIdx].z() && position[pIdx].z() <= upper.z() ){
               pexternalforce[pIdx] = bc->getForceDensity() * pmass[pIdx];
               cout << pexternalforce[pIdx] << endl;
         }
       }
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
     new_dw->put(pCrackSurfaceNormal, lb->pCrackSurfaceNormalLabel);
     new_dw->put(pCrackSurfaceContactForce, lb->pCrackSurfaceContactForceLabel);
     new_dw->put(pTensileStrength, lb->pTensileStrengthLabel);
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
				   ParticleVariable<double>& tensilestrength,
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
	            double probability;
	            double x;
                    double tensileStrengthAve = ( obj->getTensileStrengthMin() + 
                                 obj->getTensileStrengthMax() )/2;
                    double tensileStrengthWid = ( obj->getTensileStrengthMax() - 
                                 obj->getTensileStrengthMin() )/2 *
                                 obj->getTensileStrengthVariation();
	            double s;
		    do {
	              double rand = drand48();
	              s = (1-rand) * obj->getTensileStrengthMin() + 
		          rand * obj->getTensileStrengthMax();
	              
	              probability = drand48();
	              x = (s-tensileStrengthAve)/tensileStrengthWid;
	            } while( exp(-x*x) < probability );
	            tensilestrength[start+count] = s;
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


// $Log$
// Revision 1.53  2000/11/22 01:40:59  guilkey
// Moved forward declaration of GeometryPiece
//
// Revision 1.51  2000/09/25 20:23:19  sparker
// Quiet g++ warnings
//
// Revision 1.50  2000/09/22 07:10:57  tan
// MPM code works with fracture in three point bending.
//
// Revision 1.49  2000/09/07 00:38:00  tan
// Fixed a bug in ForceBC.
//
// Revision 1.48  2000/09/05 19:36:36  tan
// Fracture starts to run in Uintah/MPM!
//
// Revision 1.47  2000/09/05 07:45:13  tan
// Applied BrokenCellShapeFunction to constitutive models where fracture
// is involved.
//
// Revision 1.46  2000/09/05 05:14:58  tan
// Moved Fracture Model to MPMMaterial class.
//
// Revision 1.45  2000/08/18 20:30:52  tan
// Fixed some bugs in SerialMPM, mainly in applyPhysicalBC.
//
// Revision 1.44  2000/08/09 03:18:00  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.43  2000/08/08 04:43:13  tan
// For the default initial state, we set particle crack surface normal
// to be zero.
//
// Revision 1.42  2000/08/04 16:51:36  tan
// The "if(MPMPhysicalModules::heatConductionModel)" is removed.  Simulations
// are thermal-mechanical.
//
// Revision 1.41  2000/07/25 19:10:26  guilkey
// Changed code relating to particle combustion as well as the
// heat conduction.
//
// Revision 1.40  2000/07/20 19:45:12  guilkey
// Commented out requirement for a heat transfer coefficient as the
// current ThermalContact class doesn't use it.
//
// Revision 1.39  2000/07/05 23:43:34  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.38  2000/06/26 18:44:37  tan
// Different heat_conduction properties for different materials are allowed
// in the MPM simulation.
//
// Revision 1.37  2000/06/23 22:09:36  tan
// Added DatawareHouse allocation for ptemperatureGradient and pexternalHeatRate.
//
// Revision 1.36  2000/06/23 21:44:12  tan
// Initialize particle data of ptemperatureGradient and pexternalHeatRate for
// heat conduction.
//
// Revision 1.35  2000/06/23 01:26:16  tan
// Moved material property toughness to Fracture class.
//
// Revision 1.34  2000/06/22 22:37:47  tan
// Moved heat conduction physical parameters (thermalConductivity, specificHeat,
// and heatTransferCoefficient) from MPMMaterial class to HeatConduction class.
//
// Revision 1.33  2000/06/17 07:06:36  sparker
// Changed ProcessorContext to ProcessorGroup
//
// Revision 1.32  2000/06/15 21:57:06  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.31  2000/06/09 21:02:40  jas
// Added code to get the fudge factor directly into the constitutive model
// inititialization.
//
// Revision 1.30  2000/06/08 16:50:52  guilkey
// Changed some of the dependencies to account for what goes on in
// the burn models.
//
// Revision 1.29  2000/06/05 19:48:58  guilkey
// Added Particle IDs.  Also created NAPID (Next Available Particle ID)
// on a per patch basis so that any newly created particles will know where
// the indexing left off.
//
// Revision 1.28  2000/06/02 22:51:55  jas
// Added infrastructure for Burn models.
//
// Revision 1.27  2000/06/02 21:17:28  guilkey
// Added ParticleID's.  This isn't quite done yet, but shouldn't
// cause anything else to not work.  It will be completed ASAP.
//
// Revision 1.26  2000/06/02 17:26:36  guilkey
// Removed VarLabels from the constructor.  Now using the MPMLabel class
// instead.
//
// Revision 1.25  2000/05/31 23:54:09  rawat
// Sorry about changing getHeatTransferCoefficient, I've changed it back to original
//
// Revision 1.24  2000/05/31 23:44:54  rawat
// modified arches and properties
//
// Revision 1.23  2000/05/31 21:01:40  tan
// Added getHeatTransferCoefficient() to retrieve the material
// constants for heat exchange.
//
// Revision 1.22  2000/05/31 16:35:07  guilkey
// Added code to initialize particle temperatures.  Moved the specification
// of the temperature from the Material level to the GeometryObject level.
//
// Revision 1.21  2000/05/30 20:19:04  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.20  2000/05/26 01:43:41  tan
// Added getThermalConductivity() and getSpecificHeat()
// for computation on heat conduction.
//
