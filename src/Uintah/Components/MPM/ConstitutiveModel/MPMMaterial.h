#ifndef __MPM_MATERIAL_H__
#define __MPM_MATERIAL_H__

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/Material.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/PerPatch.h>
#include <vector>

namespace SCICore {
   namespace Geometry {
      class Point;
      class Vector;
   }
}

namespace Uintah {
   class Patch;
   class VarLabel;
   namespace MPM {
      class GeometryObject;
      class GeometryPiece;
      class ConstitutiveModel;
      class HEBurn;
      using SCICore::Geometry::Point;
      using SCICore::Geometry::Vector;
      
/**************************************
     
CLASS
   MPMMaterial

   Short description...

GENERAL INFORMATION

   MPMMaterial.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

   Copyright (C) 2000 SCI Group

KEYWORDS
   MPM_Material

DESCRIPTION
   Long description...

WARNING

****************************************/

      class MPMMaterial : public Material {
      public:
	 MPMMaterial(ProblemSpecP&);
	 
	 ~MPMMaterial();
	 
	 //////////
	 // Return correct constitutive model pointer for this material
	 ConstitutiveModel * getConstitutiveModel();
	 
	 particleIndex countParticles(const Patch*) const;
	 particleIndex countParticles(GeometryObject* obj,
				      const Patch*) const;
	 void createParticles(particleIndex numParticles,
			      PerPatch<long> NAPID,
			      const Patch*,
			      DataWarehouseP& new_dw);

	 particleIndex createParticles(GeometryObject* obj,
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
				       const Patch*);

	 int checkForSurface(const GeometryPiece* piece,
				const Point p, const Vector dxpp);
				
	 //tan: material constants for heat conduction
	 double  getThermalConductivity() const;
	 double  getSpecificHeat() const;
	 double  getHeatTransferCoefficient() const;

      private:
	 
	 // Specific constitutive model associated with this material
	 ConstitutiveModel *d_cm;

         // Burn model
	 HEBurn *d_burn;
	 
	 double d_density;
	 double d_toughness;
	 double d_temp;
	 std::vector<GeometryObject*> d_geom_objs;

	 //tan: material constants for heat conduction
	 double d_thermal_cond;
	 double d_spec_heat;
         double d_heatTransferCoefficient;
         
	 // Prevent copying of this class
	 // copy constructor
	 MPMMaterial(const MPMMaterial &mpmm);
	 MPMMaterial& operator=(const MPMMaterial &mpmm);
      };

} // end namespace MPM
} // end namespace Uintah

#endif // __MPM_MATERIAL_H__

// $Log$
// Revision 1.22  2000/06/05 19:48:58  guilkey
// Added Particle IDs.  Also created NAPID (Next Available Particle ID)
// on a per patch basis so that any newly created particles will know where
// the indexing left off.
//
// Revision 1.21  2000/06/02 22:51:55  jas
// Added infrastructure for Burn models.
//
// Revision 1.20  2000/06/02 21:17:28  guilkey
// Added ParticleID's.  This isn't quite done yet, but shouldn't
// cause anything else to not work.  It will be completed ASAP.
//
// Revision 1.19  2000/06/02 17:26:36  guilkey
// Removed VarLabels from the constructor.  Now using the MPMLabel class
// instead.
//
// Revision 1.18  2000/05/31 22:01:46  tan
// Added d_heatTransferCoefficient for heat exchange on thermal contact.
//
// Revision 1.17  2000/05/31 21:01:22  tan
// Added getHeatTransferCoefficient() to retrieve the material
// constants for heat exchange.
//
// Revision 1.16  2000/05/31 16:35:07  guilkey
// Added code to initialize particle temperatures.  Moved the specification
// of the temperature from the Material level to the GeometryObject level.
//
// Revision 1.15  2000/05/30 20:19:05  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.14  2000/05/26 01:43:29  tan
// Added getThermalConductivity() and getSpecificHeat()
// for computation on heat conduction.
//
// Revision 1.13  2000/05/19 23:15:52  guilkey
// Added code to fish out which particles are on the surface.  This
// is stored in the DW as an int 1 (it is a surface particle) or 0
// (it isn't).  Tried to do this with bools but got some stupid compiler
// error.  This will hopefully be used in conjunction with the HE burn
// models that we're going to implement.
//
// Revision 1.12  2000/05/02 06:07:12  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.11  2000/05/01 17:10:28  jas
// Added allocations for mass and volume.
//
// Revision 1.10  2000/05/01 16:18:12  sparker
// Completed more of datawarehouse
// Initial more of MPM data
// Changed constitutive model for bar
//
// Revision 1.9  2000/04/28 21:05:45  jas
// Cleaned up the creation of a material so it follows the order that is
// specified in the input file.  Also added the temp field to the MPMMaterial
// class.  All that is left to do is to read in the boundary conditions.
//
// Revision 1.8  2000/04/28 07:35:28  sparker
// Started implementation of DataWarehouse
// MPM particle initialization now works
//
// Revision 1.7  2000/04/27 23:18:44  sparker
// Added problem initialization for MPM
//
// Revision 1.6  2000/04/26 06:48:17  sparker
// Streamlined namespaces
//
// Revision 1.5  2000/04/24 21:04:26  sparker
// Working on MPM problem setup and object creation
//
// Revision 1.4  2000/04/20 18:56:18  sparker
// Updates to MPM
//
// Revision 1.3  2000/04/14 02:19:42  jas
// Now using the ProblemSpec for input.
//
// Revision 1.2  2000/03/30 18:31:22  guilkey
// Moved Material base class to Grid directory.  Modified MPMMaterial
// and sub.mk to coincide with these changes.
//
// Revision 1.1  2000/03/24 00:45:43  guilkey
// Added MPMMaterial class, as well as a skeleton Material class, from
// which MPMMaterial is inherited.  The Material class will be filled in
// as it's mission becomes better identified.
//

