#ifndef __CONSTITUTIVE_MODEL_H__
#define __CONSTITUTIVE_MODEL_H__

namespace Uintah {
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

class ConstitutiveModel {
public:

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

