#ifndef __GUV_CONSTITUTIVE_MODEL_H__
#define __GUV_CONSTITUTIVE_MODEL_H__

#include <math.h>
#include "ShellMaterial.h"
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

////////////////////////////////////////////////////////////////////////////
/*! 
   \class GUVMaterial
   \brief Material model for Giant Unilamellar Vescicles
   \author Biswajit Banerjee \n
   C-SAFE and Department of Mechanical Engineering \n
   University of Utah \n
   Copyright (C) 2004 University of Utah \n

   The GUVs are assumed to consist of two phases that have different 
   shear and bulk moduli.

   \warning  Only isotropic hyperelastic materials
*/
////////////////////////////////////////////////////////////////////////////

  class GUVMaterial : public ShellMaterial {

  public:

    /*! Datatype for storing model parameters */
    struct CMDataGUV {
      double Bulk_lipid;
      double Shear_lipid;
      double Bulk_cholesterol;
      double Shear_cholesterol;
    };

  private:
    CMDataGUV d_cm;

    // Prevent copying of this class
    // copy constructor
    //GUVMaterial(const GUVMaterial &cm);
    GUVMaterial& operator=(const GUVMaterial &cm);

    enum MaterialType {
      Lipid = 0,
      Cholesterol
    };

  public:
    // constructors
    GUVMaterial(ProblemSpecP& ps,  MPMLabel* lb, int n8or27);
    GUVMaterial(const GUVMaterial* cm);
       
    // destructor
    virtual ~GUVMaterial();

    /*! Schedule compute of type - the rest come from ShellMaterial */
    virtual void addInitialComputesAndRequires(Task* task,
					       const MPMMaterial* matl,
					       const PatchSet* patches) const;

    /*! initialize each GUV particle's constitutive model data */
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

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);
         
    // compute stable timestep for this patch
    virtual void computeStableTimestep(const Patch* patch,
				       const MPMMaterial* matl,
				       DataWarehouse* new_dw);

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const PatchSubset* patches,
				     const MPMMaterial* matl,
				     DataWarehouse* old_dw,
				     DataWarehouse* new_dw);

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Add computes and requires update of rotation rate */
    //
    virtual void addComputesRequiresRotRateUpdate(Task* task,
                                                  const MPMMaterial* matl,
                                                  const PatchSet* patches); 

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Actually update rotation rate */
    //
    virtual void particleNormalRotRateUpdate(const PatchSubset* patches,
				             const MPMMaterial* matl,
				             DataWarehouse* old_dw,
				             DataWarehouse* new_dw);

    virtual double computeRhoMicroCM(double pressure,
				     const double p_ref,
				     const MPMMaterial* matl);

    virtual void computePressEOSCM(double rho_m, double& press_eos,
				   double p_ref,
				   double& dp_drho, double& ss_new,
				   const MPMMaterial* matl);

    virtual double getCompressibility();

  private:

    ///////////////////////////////////////////////////////////////////////////
    //
    /*! Calculate the plane stress deformation gradient corresponding
    // to sig33 = 0 and the Cauchy stress */
    //
    virtual bool computePlaneStressAndDefGrad(Matrix3& F, Matrix3& sig,
                                      double bulk, double shear);

  };
} // End namespace Uintah
      


#endif  // __GUV_CONSTITUTIVE_MODEL_H__ 

