//  CompMooneyRivlin.h 

#ifndef __COMPMOONRIV_CONSTITUTIVE_MODEL_H__
#define __COMPMOONRIV_CONSTITUTIVE_MODEL_H__


#include "ConstitutiveModel.h"	
#include <math.h>
#include "../Util/Matrix3.h"

class CompMooneyRivlin : public ConstitutiveModel {
 private:

  // Create datatype for storing model parameters
  struct CMData {
        double C1;
        double C2;
        double C3;
        double C4;
        };

  // Prevent copying of this class
  // copy constructor
  CompMooneyRivlin(const CompMooneyRivlin &cm);
  CompMooneyRivlin& operator=(const CompMooneyRivlin &cm);
 
 public:
  // constructors
  CompMooneyRivlin(const Region* region,
                   const MPMMaterial* matl,
                   const DataWarehouseP& old_dw,
                   DataWarehouseP& new_dw);

  // destructor 
  virtual ~CompMooneyRivlin();

  // compute stress at each particle in the region
  virtual void computeStressTensor(const Region* region,
				   const MPMMaterial* matl,
                                   const DataWarehouseP& old_dw,
                                   DataWarehouseP& new_dw);

  // compute total strain energy for all particles in the region
  virtual double computeStrainEnergy(const Region* region,
                                     const MPMMaterial* matl,
                                     const DataWarehouseP& new_dw);

  // initialize  each particle's constitutive model data
  virtual void intitializeCMData(const Region* region,
                            const MPMMaterial* matl,
                            DataWarehouseP& new_dw);

};

#endif  // __COMPMOONRIV_CONSTITUTIVE_MODEL_H__ 

// $Log$
// Revision 1.2  2000/03/15 20:05:56  guilkey
// Worked over the ConstitutiveModel base class, and the CompMooneyRivlin
// class to operate on all particles in a region of that material type at once,
// rather than on one particle at a time.  These changes will require some
// improvements to the DataWarehouse before compilation will be possible.
//
// Revision 1.1  2000/03/14 22:11:47  jas
// Initial creation of the constitutive model directory with the legacy
// constitutive model classes.
//
