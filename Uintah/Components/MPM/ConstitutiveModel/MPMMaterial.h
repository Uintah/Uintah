#ifndef __MPM_MATERIAL_H__
#define __MPM_MATERIAL_H__

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Grid/Material.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <vector>

namespace SCICore {
   namespace Geometry {
      class Point;
      class Vector;
   }
}

namespace Uintah {
   class Region;
   class VarLabel;
   namespace MPM {
      class GeometryObject;
      class GeometryPiece;
      class ConstitutiveModel;
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
	 
	 particleIndex countParticles(const Region*) const;
	 particleIndex countParticles(GeometryObject* obj,
				      const Region*) const;
	 void createParticles(particleIndex numParticles,
			      const Region*,
			      DataWarehouseP& new_dw);
	 particleIndex createParticles(GeometryObject* obj,
				       particleIndex start,
				       ParticleVariable<Point>& position,
				       ParticleVariable<Vector>& velocity,
				       ParticleVariable<Vector>& pexternalforce,
				       ParticleVariable<double>& mass,
				       ParticleVariable<double>& volume,
				       ParticleVariable<int>& pissurf,
				       const Region*);

	 int checkForSurface(const GeometryPiece* piece,
				const Point p, const Vector dxpp);
				
	 double  getThermalConductivity() const;
	 double  getSpecificHeat() const;

      private:
	 
	 // Specific constitutive model associated with this material
	 ConstitutiveModel *d_cm;
	 
	 double d_density;
	 double d_toughness;
	 double d_thermal_cond;
	 double d_spec_heat;
	 double d_temp;
	 std::vector<GeometryObject*> d_geom_objs;
	 
	 // Prevent copying of this class
	 // copy constructor
	 MPMMaterial(const MPMMaterial &mpmm);
	 MPMMaterial& operator=(const MPMMaterial &mpmm);
	 
	 const VarLabel* pDeformationMeasureLabel;
	 const VarLabel* pStressLabel;
	 const VarLabel* pVolumeLabel;
	 const VarLabel* pMassLabel;
	 const VarLabel* pVelocityLabel;
	 const VarLabel* pExternalForceLabel;
	 const VarLabel* pXLabel;
	 const VarLabel* pSurfLabel;
      };

} // end namespace MPM
} // end namespace Uintah

#endif // __MPM_MATERIAL_H__

// $Log$
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

