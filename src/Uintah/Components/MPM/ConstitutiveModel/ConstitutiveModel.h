
#ifndef __CONSTITUTIVE_MODEL_H__
#define __CONSTITUTIVE_MODEL_H__

class ConstitutiveModel {
public:

  // Basic constitutive model calculations
  virtual void computeStressTensor(const Region* region,
				   const MPMMaterial* matl,
				   const DataWarehouseP& new_dw,
				   DataWarehouseP& old_dw) = 0;

  // Computation of strain energy.  Useful for tracking energy balance.
  virtual double computeStrainEnergy(const Region* region,
				   const MPMMaterial* matl,
                                   const DataWarehouseP& new_dw) = 0;

  // Create space in data warehouse for CM data
  virtual void initializeCMData(const Region* region,
				const MPMMaterial* matl,
                                DataWarehouseP& new_dw) = 0;

};

#endif  // __CONSTITUTIVE_MODEL_H__

// $Log$
// Revision 1.3  2000/03/15 20:05:56  guilkey
// Worked over the ConstitutiveModel base class, and the CompMooneyRivlin
// class to operate on all particles in a region of that material type at once,
// rather than on one particle at a time.  These changes will require some
// improvements to the DataWarehouse before compilation will be possible.
//
