#ifndef __CNHDAMAGE_MODEL_H__
#define __CNHDAMAGE_MODEL_H__


#include "CompNeoHook.h"        

namespace Uintah {

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class CNHDamage
    \brief Compressible Neo-Hookean Elastic Material with Damage
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2004 University of Utah
  */
  /////////////////////////////////////////////////////////////////////////////

  class CNHDamage : public CompNeoHook {

  public:

    // Create datatype for failure strains
    struct FailureStrainData {
      double mean;      /*< Mean failure strain */
      double std;       /*< Standard deviation of failure strain */
      std::string dist; /*< Failure strain distrinution */
    };

    const VarLabel* pFailureStrainLabel;
    const VarLabel* pFailedLabel;
    const VarLabel* pDeformRateLabel;
    const VarLabel* pFailureStrainLabel_preReloc;
    const VarLabel* pFailedLabel_preReloc;
    const VarLabel* pDeformRateLabel_preReloc;

  protected:

    FailureStrainData d_epsf;

    // Erosion algorithms
    bool d_setStressToZero;
    bool d_allowNoTension;
    bool d_removeMass;

  private:

    // Prevent assignment of objects of this class
    CNHDamage& operator=(const CNHDamage &cm);

  public:

    // constructors
    CNHDamage(ProblemSpecP& ps,  MPMLabel* lb, MPMFlags* flag);
    CNHDamage(const CNHDamage* cm);
       
    // destructor
    virtual ~CNHDamage();

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

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Add the requires for failure simulation. */
    ////////////////////////////////////////////////////////////////////////
    virtual void addRequiresDamageParameter(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches) const;


    ////////////////////////////////////////////////////////////////////////
    /*! \brief Get the flag that marks a failed particle. */
    ////////////////////////////////////////////////////////////////////////
    virtual void getDamageParameter(const Patch* patch, 
                                    ParticleVariable<int>& damage, int dwi,
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

    void getFailureStrainData(ProblemSpecP& ps);

    void setFailureStrainData(const CNHDamage* cm);

    void initializeLocalMPMLabels();

    void setErosionAlgorithm();

    void setErosionAlgorithm(const CNHDamage* cm);

  protected:

    // Modify the stress if particle has failed
    void updateFailedParticlesAndModifyStress(const Matrix3& bb, 
                                              const double& pFailureStrain, 
                                              const int& pFailed,
                                              int& pFailed_new, 
                                              Matrix3& pStress_new);

    // compute stress at each particle in the patch
    void computeStressTensorImplicit(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

    /*! Compute tangent stiffness matrix */
    void computeTangentStiffnessMatrix(const Matrix3& sigDev, 
                                       const double&  mubar,
                                       const double&  J,
                                       const double&  bulk,
                                       double D[6][6]);
    /*! Compute BT*Sig*B (KGeo) */
    void BnlTSigBnl(const Matrix3& sig, const double Bnl[3][24],
                    double BnTsigBn[24][24]) const;

    /*! Compute K matrix */
    void computeStiffnessMatrix(const double B[6][24],
                                const double Bnl[3][24],
                                const double D[6][6],
                                const Matrix3& sig,
                                const double& vol_old,
                                const double& vol_new,
                                double Kmatrix[24][24]);
  };
} // End namespace Uintah
      
#endif  // __CNHDAMAGE_MODEL_H__ 

