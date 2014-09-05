//  Water.h 
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//    This is for a model for fluid as given by Eqn. 64 in
//    IJNME, 60:1475-1512, 2004 "On the Galerkin formulation of the SPH method"
//    L. Cueto-Felgueroso, et al.
//    Features:
//      Usage:



#ifndef __WATER_CONSTITUTIVE_MODEL_H__
#define __WATER_CONSTITUTIVE_MODEL_H__


#include <math.h>
#include "ConstitutiveModel.h"  
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

namespace Uintah {
  class Water : public ConstitutiveModel {

  public:

    // Create datatype for storing model parameters
    struct CMData {
      double d_Bulk;
      double d_Viscosity;
      double d_Gamma;
    };

  protected:

    CMData d_initialData;
    bool d_useModifiedEOS; 
    int d_8or27;

  private:
    // Prevent copying of this class
    // copy constructor
    //Water(const Water &cm);
    Water& operator=(const Water &cm);

  public:
    // constructors
    Water(ProblemSpecP& ps, MPMFlags* flag);
    Water(const Water* cm);
       
    // destructor
    virtual ~Water();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone
    Water* clone();
    
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

    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb) const;


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

  };
} // End namespace Uintah

#endif  // __WATER_CONSTITUTIVE_MODEL_H__ 

