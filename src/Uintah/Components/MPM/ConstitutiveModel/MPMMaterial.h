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
   }
}

namespace Uintah {
   class Region;
   namespace MPM {
      class GeometryObject;
      class ConstitutiveModel;
      using SCICore::Geometry::Point;
      
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
	 
	 long countParticles(const Region*) const;
	 long countParticles(GeometryObject* obj, const Region*) const;
	 void createParticles(ParticleVariable<Point>& position, const Region*);
	 long createParticles(GeometryObject* obj, long start,
			      ParticleVariable<Point>& position, const Region*);
      private:
	 
	 // Specific constitutive model associated with this material
	 ConstitutiveModel *d_cm;
	 
	 double d_density;
	 double d_toughness;
	 double d_thermal_cond;
	 double d_spec_heat;
	 std::vector<GeometryObject*> d_geom_objs;
	 
	 // Prevent copying of this class
	 // copy constructor
	 MPMMaterial(const MPMMaterial &mpmm);
	 MPMMaterial& operator=(const MPMMaterial &mpmm);
	 
      };

} // end namespace MPM
} // end namespace Uintah

#endif // __MPM_MATERIAL_H__

// $Log$
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

