//  HypoElastic.h 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for HypoElasticity
//    Features:
//      Usage:



#ifndef __HYPOELASTIC_CONSTITUTIVE_MODEL_H__
#define __HYPOELASTIC_CONSTITUTIVE_MODEL_H__


#include <math.h>
#include "ConstitutiveModel.h"	
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  class HypoElastic : public ConstitutiveModel {
  private:
    // Create datatype for storing model parameters
  public:
    struct CMData {
      double G;
      double K;
    };

  private:
    friend const TypeDescription* fun_getTypeDescription(CMData*);

    CMData d_initialData;
    //double d_se;
    // Prevent copying of this class
    // copy constructor
    HypoElastic(const HypoElastic &cm);
    HypoElastic& operator=(const HypoElastic &cm);

  public:
    // constructors
    HypoElastic(ProblemSpecP& ps, MPMLabel* lb, int n8or27);
       
    // destructor
    virtual ~HypoElastic();
    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
				       const MPMMaterial* matl,
				       DataWarehouse* new_dw);

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
				     const MPMMaterial* matl,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw);

    // carry forward CM data for RigidMPM
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

    // initialize  each particle's constitutive model data
    virtual void initializeCMData(const Patch* patch,
				  const MPMMaterial* matl,
				  DataWarehouse* new_dw);


    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
				   ParticleSubset* subset,
				   map<const VarLabel*, ParticleVariableBase*>* newState,
				   ParticleSubset* delset,
				   DataWarehouse* old_dw);


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

    // Convert J-integral into stress intensity factors for hypoelastic materials
    virtual void ConvertJToK(const MPMMaterial* matl, const Vector& J,
                             const Vector& C,const Vector& V,
                             Vector& SIF);
  };

} // End namespace Uintah

#endif  // __HYPOELASTIC_CONSTITUTIVE_MODEL_H__ 
