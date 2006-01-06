//  TransIsoHyper.h
//  class ConstitutiveModel ConstitutiveModel data type -- 3D - 
//  holds ConstitutiveModel
//  information for the FLIP technique:
//    This is for a Transversely Isotropic Hyperelastic material
//    Features:
//      Usage:



#ifndef __TransIsoHyper_CONSTITUTIVE_MODEL_H__
#define __TransIsoHyper_CONSTITUTIVE_MODEL_H__


#include <math.h>
#include "ConstitutiveModel.h"	
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

namespace Uintah {
  class TransIsoHyper : public ConstitutiveModel {
  private:
    // Create datatype for storing model parameters
    bool d_useModifiedEOS;
  public:
    struct CMData {   //_________________________________________modified here
      double Bulk;
      Vector a0;
      double c1;
      double c2;
      double c3;
      double c4;
      double c5;
      double lambda_star;
      double failure;
      double crit_shear;
      double crit_stretch;
    };

    const VarLabel* pStretchLabel;  // For diagnostic
    const VarLabel* pStretchLabel_preReloc;  // For diagnostic
    
    const VarLabel* pFailureLabel;  // ____________________________fail_labels
    const VarLabel* pFailureLabel_preReloc;

  private:
    CMData d_initialData;

    // Prevent copying of this class
    // copy constructor
    //TransIsoHyper(const TransIsoHyper &cm);
    TransIsoHyper& operator=(const TransIsoHyper &cm);
    int d_8or27;

  public:
    // constructors
    TransIsoHyper(ProblemSpecP& ps,  MPMLabel* lb, MPMFlags* flag);
    TransIsoHyper(const TransIsoHyper* cm);
       
    // destructor
    virtual ~TransIsoHyper();

    // clone
    TransIsoHyper* clone();

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

    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet*) const;

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

    virtual Vector getInitialFiberDir();

    virtual void addParticleState(std::vector<const VarLabel*>& from,
				  std::vector<const VarLabel*>& to);

  };
} // End namespace Uintah
      


#endif  // __TransIsoHyper_CONSTITUTIVE_MODEL_H__ 

