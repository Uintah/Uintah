//  MWViscoElastic.h 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for MWViscoElastic
//    Features:
//      Usage:



#ifndef __MWVISCOELASTIC_CONSTITUTIVE_MODEL_H__
#define __MWVISCOELASTIC_CONSTITUTIVE_MODEL_H__


#include <math.h>
#include "ConstitutiveModel.h"	
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class MWViscoElastic : public ConstitutiveModel {
  private:
    // Create datatype for storing model parameters
  public:
    struct CMData {
      double E_Shear;
      double E_Bulk;
      double VE_Shear;
      double VE_Bulk;
      double V_Viscosity;
      double D_Viscosity;
    };

  private:
    friend const TypeDescription* fun_getTypeDescription(CMData*);

    CMData d_initialData;
    double d_se;
    // Prevent copying of this class
    // copy constructor
    MWViscoElastic(const MWViscoElastic &cm);
    MWViscoElastic& operator=(const MWViscoElastic &cm);

  public:
    // constructors
    MWViscoElastic(ProblemSpecP& ps, MPMLabel* lb, int n8or27);
       
    // destructor
    virtual ~MWViscoElastic();
    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
				       const MPMMaterial* matl,
				       DataWarehouse* new_dw);

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
				     const MPMMaterial* matl,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw);

    // initialize  each particle's constitutive model data
    virtual void initializeCMData(const Patch* patch,
				  const MPMMaterial* matl,
				  DataWarehouse* new_dw);

    virtual void allocateCMData(DataWarehouse* new_dw,
				ParticleSubset* subset,
				map<const VarLabel*, ParticleVariableBase*>* newState);


    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches) const;

    virtual void addComputesAndRequires(Task* task,
					const MPMMaterial* matl,
					const PatchSet* patches,
					const bool recursion) const;

    virtual double computeRhoMicroCM(double pressure,
				     const double p_ref,
				     const MPMMaterial* matl);

    virtual void computePressEOSCM(double rho_m, double& press_eos,
				   double p_ref,
				   double& dp_drho, double& ss_new,
				   const MPMMaterial* matl);

    virtual double getCompressibility();


    virtual void addParticleState(std::vector<const VarLabel*>& from,
				  std::vector<const VarLabel*>& to);
  };

} // End namespace Uintah

#endif  // __MWVISCOELASTIC_CONSTITUTIVE_MODEL_H__ 
