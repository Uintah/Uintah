//  MPMMaterial.cc
//

#include "MPMMaterial.h"
#include "ConstitutiveModel.h"
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/CellIterator.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/PerPatch.h>
#include <Uintah/Components/MPM/GeometrySpecification/GeometryPieceFactory.h>
#include <Uintah/Components/MPM/GeometrySpecification/UnionGeometryPiece.h>
#include <Uintah/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Uintah/Exceptions/ParameterNotFound.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <iostream>
#include "ConstitutiveModelFactory.h"
#include <Uintah/Components/MPM/Burn/HEBurnFactory.h>
#include <Uintah/Components/MPM/Burn/HEBurn.h>
#include <Uintah/Components/MPM/MPMLabel.h>
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
   std::cerr << "works here after cm factory" << std::endl;

   d_burn = HEBurnFactory::create(ps);
   if (!d_burn)
	throw ParameterNotFound("No burn model");

   // Step 2 -- get the general material properties

   ps->require("density",d_density);
   ps->require("toughness",d_toughness);
   ps->require("thermal_conductivity",d_thermal_cond);
   ps->require("specific_heat",d_spec_heat);

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
      cerr << "piece: " << piece_num << '\n';
      d_geom_objs.push_back(scinew GeometryObject(mainpiece, geom_obj_ps));

      // Step 4 -- Assign the boundary conditions to the object

      
      // Step 5 -- Assign the velocity field
      int vf;
      ps->require("velocity_field",vf);
      setVFIndex(vf);

   }

}

MPMMaterial::~MPMMaterial()
{
  // Destructor
}

ConstitutiveModel * MPMMaterial::getConstitutiveModel()
{
  // Return the pointer to the constitutive model associated
  // with this material

  return d_cm;
}

HEBurn * MPMMaterial::getBurnModel()
{
  // Return the pointer to the constitutive model associated
  // with this material

  return d_burn;
}

particleIndex MPMMaterial::countParticles(const Patch* patch) const
{
   particleIndex sum = 0;
   for(int i=0; i<d_geom_objs.size(); i++)
      sum+= countParticles(d_geom_objs[i], patch);
   return sum;
}

void MPMMaterial::createParticles(particleIndex numParticles,
				  PerPatch<long> NAPID,
				  const Patch* patch,
				  DataWarehouseP& new_dw)
{

   const MPMLabel* lb = MPMLabel::getLabels();

   ParticleVariable<Point> position;
   new_dw->allocate(numParticles, position, lb->pXLabel,
		    getDWIndex(), patch);
   ParticleVariable<Vector> pvelocity;
   new_dw->allocate(pvelocity, lb->pVelocityLabel, getDWIndex(), patch);
   ParticleVariable<Vector> pexternalforce;
   new_dw->allocate(pexternalforce, lb->pExternalForceLabel,
					getDWIndex(), patch);
   ParticleVariable<double> pmass;
   new_dw->allocate(pmass, lb->pMassLabel, getDWIndex(), patch);
   ParticleVariable<double> pvolume;
   new_dw->allocate(pvolume, lb->pVolumeLabel, getDWIndex(), patch);
   ParticleVariable<int> pissurf;
   new_dw->allocate(pissurf, lb->pSurfLabel, getDWIndex(), patch);
   ParticleVariable<double> ptemperature;
   new_dw->allocate(ptemperature, lb->pTemperatureLabel, getDWIndex(), patch);
   ParticleVariable<long> pparticleID;
   new_dw->allocate(pparticleID, lb->pParticleIDLabel, getDWIndex(), patch);

   particleIndex start = 0;
   for(int i=0; i<d_geom_objs.size(); i++){
      start += createParticles( d_geom_objs[i], start, position,
				pvelocity,pexternalforce,pmass,pvolume,
				pissurf,ptemperature,pparticleID,NAPID,patch);
   }

   new_dw->put(position, lb->pXLabel, getDWIndex(), patch);
   new_dw->put(pvelocity, lb->pVelocityLabel, getDWIndex(), patch);
   new_dw->put(pexternalforce, lb->pExternalForceLabel, getDWIndex(), patch);
   new_dw->put(pmass, lb->pMassLabel, getDWIndex(), patch);
   new_dw->put(pvolume, lb->pVolumeLabel, getDWIndex(), patch);
   new_dw->put(pissurf, lb->pSurfLabel, getDWIndex(), patch);
   new_dw->put(ptemperature, lb->pTemperatureLabel, getDWIndex(), patch);
   new_dw->put(pparticleID, lb->pParticleIDLabel, getDWIndex(), patch);

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
	       if(piece->inside(p))
		  count++;
	    }
	 }
      }
   }
   cerr << "Count1 for obj: " << count << '\n';
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
		  pissurf[start+count]=checkForSurface(piece,p,dxpp);
		  pexternalforce[start+count]=Vector(0,0,0); // for now
		  particleID[start+count]=
				(patch_number | (NAPID + start + count));
		  count++;
	       }
	    }
	 }
      }
   }
   cerr << "Count2 for obj: " << count << '\n';
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

double  MPMMaterial::getThermalConductivity() const
{
  return d_thermal_cond;
}

double  MPMMaterial::getSpecificHeat() const
{
  return d_spec_heat;
}

double  MPMMaterial::getHeatTransferCoefficient() const
{
  return d_heatTransferCoefficient;
}

// $Log$
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
