#ifndef __CONSTITUTIVE_MODEL_H__
#define __CONSTITUTIVE_MODEL_H__

#include <Uintah/Interface/DataWarehouseP.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <Uintah/Interface/ProblemSpec.h>

using Uintah::Interface::ProblemSpecP;
using Uintah::Interface::ProblemSpec;

namespace Uintah {
    namespace Grid {
	class Region;
    }
namespace Components {

    // THIS DOES NOT GO HERE - steve
class MPMMaterial;

/**************************************

CLASS
   ConstitutiveModel
   
   Short description...

GENERAL INFORMATION

   ConstitutiveModel.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Constitutive_Model

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

using Uintah::Grid::Region;
using Uintah::Interface::DataWarehouseP;

class ConstitutiveModel {
public:

  ConstitutiveModel();
  virtual ~ConstitutiveModel();

  //////////
  // Basic constitutive model calculations
  virtual void computeStressTensor(const Region* region,
				   const MPMMaterial* matl,
				   const DataWarehouseP& new_dw,
				   DataWarehouseP& old_dw) = 0;

  //////////
  // Computation of strain energy.  Useful for tracking energy balance.
  virtual double computeStrainEnergy(const Region* region,
				   const MPMMaterial* matl,
                                   const DataWarehouseP& new_dw) = 0;

  //////////
  // Create space in data warehouse for CM data
  virtual void initializeCMData(const Region* region,
				const MPMMaterial* matl,
                                DataWarehouseP& new_dw) = 0;

};

} // end namespace Components
} // end namespace Uintah

// $Log$
// Revision 1.7  2000/04/14 02:19:41  jas
// Now using the ProblemSpec for input.
//
// Revision 1.6  2000/03/20 17:17:08  sparker
// Made it compile.  There are now several #idef WONT_COMPILE_YET statements.
//
// Revision 1.5  2000/03/17 09:29:34  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.4  2000/03/17 02:57:02  dav
// more namespace, cocoon, etc
//
// Revision 1.3  2000/03/15 20:05:56  guilkey
// Worked over the ConstitutiveModel base class, and the CompMooneyRivlin
// class to operate on all particles in a region of that material type at once,
// rather than on one particle at a time.  These changes will require some
// improvements to the DataWarehouse before compilation will be possible.
//

#endif  // __CONSTITUTIVE_MODEL_H__

