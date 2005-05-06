#ifndef __RIGID_CONSTITUTIVE_MODEL_H__
#define __RIGID_CONSTITUTIVE_MODEL_H__

#include "ConstitutiveModel.h"  
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  class MPMLabel;
  class MPMFlags;

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class RigidMaterial
    \brief Rigid material - no stresses or deformation.
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2000 University of Utah

    The material does not deform and does not develop any stresses or
    internal heating.

    Shear and bulk moduli are used to compute the wave speed and contact
    and interaction with ICE.
  */
  /////////////////////////////////////////////////////////////////////////////

  class RigidMaterial : public ConstitutiveModel {

  public:
    struct CMData {
      double G;
      double K;
    };

  private:

    CMData d_initialData;

    // Prevent assignment of this class
    RigidMaterial& operator=(const RigidMaterial &cm);

  public:

    // constructors
    RigidMaterial(ProblemSpecP& ps, MPMLabel* lb, MPMFlags* flag);
    RigidMaterial(const RigidMaterial* cm);
       
    // destructor
    virtual ~RigidMaterial();

    /*! initialize  each particle's constitutive model data */
    virtual void initializeCMData(const Patch* patch,
                                  const MPMMaterial* matl,
                                  DataWarehouse* new_dw);

    /*! Computes and requires for compute stress tensor added to
      the taskgraph */
    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    /*! compute stress at each particle in the patch */
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

    /* Add computes and requires for the implicit code */
    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        const bool recursion) const;

    /* Computes stress tensor for the implicit code */
    virtual void computeStressTensor(const PatchSubset* ,
                                     const MPMMaterial* ,
                                     DataWarehouse* ,
                                     DataWarehouse* ,
#ifdef HAVE_PETSC
                                     MPMPetscSolver* ,
#else
                                     SimpleSolver* ,
#endif
                                     const bool );

    /*! carry forward CM data for RigidMPM */
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

    /*! Add requires to task graph for particle conversion */
    virtual void allocateCMDataAddRequires(Task* task, 
                                           const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb) const
    {
    }

    /*! Add requires to task graph for particle conversion */
    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* subset,
                                   map<const VarLabel*, 
                                      ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw)
    {
    }

    /*! Add particle state for relocation */
    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);

    /*! Function that interacts with ice */
    virtual double computeRhoMicroCM(double pressure,
                                     const double p_ref,
                                     const MPMMaterial* matl);

    /*! Function that interacts with ice */
    virtual void computePressEOSCM(double rho_m, double& press_eos,
                                   double p_ref,
                                   double& dp_drho, double& ss_new,
                                   const MPMMaterial* matl);

    /*! Function that interacts with ice */
    virtual double getCompressibility();

  protected:

    /*! compute stress at each particle in the patch (replacement for
        standard compute stress tensor without the recursion flag) */
    void computeStressTensorImplicit(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

  };

} // End namespace Uintah

#endif  // __RIGID_CONSTITUTIVE_MODEL_H__ 
