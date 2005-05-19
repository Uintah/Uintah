#ifndef __CNHPDAMAGE_MODEL_H__
#define __CNHPDAMAGE_MODEL_H__

#include "CNHDamage.h"  

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class CNHPDamage
    \brief Compressible Neo-Hookean Elastic-Plastic Material with Damage
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2004 University of Utah
  */
  /////////////////////////////////////////////////////////////////////////////

  class CNHPDamage : public CNHDamage {

  public:

    // Create datatype for storing plasticity parameters
    struct PlasticityData {
      double FlowStress;
      double K;
    };   

    const VarLabel* pPlasticStrainLabel;
    const VarLabel* pPlasticStrainLabel_preReloc;

  protected:

    PlasticityData d_plastic;
         
  private:

    // Prevent assignment of objects of this class
    CNHPDamage& operator=(const CNHPDamage &cm);

  public:

    // constructors
    CNHPDamage(ProblemSpecP& ps, MPMLabel* lb,MPMFlags* flag);
    CNHPDamage(const CNHPDamage* cm);
         
    // destructor 
    virtual ~CNHPDamage();

    // clone
    CNHPDamage* clone();
         
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

    // initialize  each particle's constitutive model data
    virtual void initializeCMData(const Patch* patch,
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

    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* ,
                                        const bool ) const;

    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw,
#ifdef HAVE_PETSC
                                     MPMPetscSolver* solver,
#else
                                     SimpleSolver* solver,
#endif
                                     const bool recursion);

    // carry forward CM data for RigidMPM
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

    virtual void allocateCMDataAddRequires(Task* task, 
                                           const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb) const;

    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* subset,
                                   map<const VarLabel*, 
                                   ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw);

    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);

  private:

    void getPlasticityData(ProblemSpecP& ps);

    void setPlasticityData(const CNHPDamage* cm);

    void initializeLocalMPMLabels();

    // compute stress at each particle in the patch
    void computeStressTensorImplicit(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

    /*! Compute tangent stiffness matrix */
    void computeTangentStiffnessMatrix(const Matrix3& tauDevTrial, 
                                       const Matrix3& normal,
                                       const double&  mubar,
                                       const double&  delGamma,
                                       const double&  J,
                                       const double&  bulk,
                                       double D[6][6]);
  };
} // End namespace Uintah

#endif  // __CNHPDAMAGE_MODEL_H__ 
