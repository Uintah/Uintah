#ifndef __ELASTIC_PLASTIC_H__
#define __ELASTIC_PLASTIC_H__


#include "ConstitutiveModel.h"
#include "ImplicitCM.h"
#include "PlasticityModels/YieldCondition.h"
#include "PlasticityModels/StabilityCheck.h"
#include "PlasticityModels/PlasticityModel.h"
#include "PlasticityModels/DamageModel.h"
#include "PlasticityModels/MPMEquationOfState.h"
#include "PlasticityModels/ShearModulusModel.h"
#include "PlasticityModels/MeltingTempModel.h"
#include <math.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/TangentModulusTensor.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouseP.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>

namespace Uintah {

  class MPMLabel;
  class MPMFlags;

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class ElasticPlastic
    \brief High-strain rate Hypo-Elastic Plastic Constitutive Model
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n
    Copyright (C) 2002 University of Utah

    The rate of deformation and stress is rotated to material configuration 
    before the updated values are calculated.  The left stretch and rotation 
    are updated incrementatlly to get the deformation gradient.

    Needs :
    1) Isotropic elastic moduli.
    2) Flow rule in the form of a Plasticity Model.
    3) Yield condition.
    4) Stability condition.
    5) Damage model.

    \warning Only isotropic materials, von-Mises type yield conditions, 
    associated flow rule, high strain rate.
  */
  /////////////////////////////////////////////////////////////////////////////

  class ElasticPlastic : public ConstitutiveModel, public ImplicitCM {

  public:
    // Create datatype for storing model parameters
    struct CMData {
      double Bulk;    /*< Bulk modulus */
      double Shear;   /*< Shear Modulus */
      double alpha;   /*< Coeff. of thermal expansion */
      double Chi;     /*< Taylor-Quinney coefficient */
    };   

    // Create datatype for storing porosity parameters
    struct PorosityData {
      double f0;     /*< Initial mean porosity */
      double f0_std; /*< Initial standard deviation of porosity */
      double fc;     /*< Critical porosity */
      double fn;     /*< Volume fraction of void nucleating particles */
      double en;     /*< Mean strain for nucleation */
      double sn;     /*< Standard deviation of strain for nucleation */
      std::string porosityDist; /*< Initial porosity distribution*/
    };

    // Create datatype for storing damage parameters
    struct ScalarDamageData {
      double D0;     /*< Initial mean scalar damage */
      double D0_std; /*< Initial standard deviation of scalar damage */
      double Dc;     /*< Critical scalar damage */
      std::string scalarDamageDist; /*< Initial damage distrinution */
    };

    // Create a datatype for storing Cp calculation paramaters
    struct CpData {
      double A;
      double B;
      double C;
    };

    const VarLabel* pRotationLabel;  // For Hypoelastic-plasticity
    const VarLabel* pStrainRateLabel;  
    const VarLabel* pPlasticStrainLabel;  
    const VarLabel* pPlasticStrainRateLabel;  
    const VarLabel* pDamageLabel;  
    const VarLabel* pPorosityLabel;  
    const VarLabel* pLocalizedLabel;  

    const VarLabel* pRotationLabel_preReloc;  // For Hypoelastic-plasticity
    const VarLabel* pStrainRateLabel_preReloc;  
    const VarLabel* pPlasticStrainLabel_preReloc;  
    const VarLabel* pPlasticStrainRateLabel_preReloc;  
    const VarLabel* pDamageLabel_preReloc;  
    const VarLabel* pPorosityLabel_preReloc;  
    const VarLabel* pLocalizedLabel_preReloc;  

  protected:

    CMData           d_initialData;
    PorosityData     d_porosity;
    ScalarDamageData d_scalarDam;
    CpData           d_Cp;
    
    double d_tol;
    double d_initialMaterialTemperature;
    double d_isothermal;
    bool   d_useModifiedEOS;
    bool   d_evolvePorosity;
    bool   d_evolveDamage;
    bool   d_computeSpecificHeat;
    bool   d_checkTeplaFailureCriterion;
    bool   d_doMelting;

    // Erosion algorithms
    bool   d_setStressToZero;
    bool   d_allowNoTension;
    bool   d_removeMass;

    YieldCondition*     d_yield;
    StabilityCheck*     d_stable;
    PlasticityModel*    d_plastic;
    DamageModel*        d_damage;
    MPMEquationOfState* d_eos;
    ShearModulusModel*  d_shear;
    MeltingTempModel*   d_melt;
         
  private:
    // Prevent copying of this class
    // copy constructor
    //ElasticPlastic(const ElasticPlastic &cm);
    ElasticPlastic& operator=(const ElasticPlastic &cm);

  public:

    ////////////////////////////////////////////////////////////////////////
    /*! \brief constructors */
    ////////////////////////////////////////////////////////////////////////
    ElasticPlastic(ProblemSpecP& ps, MPMLabel* lb,MPMFlags* flag);
    ElasticPlastic(const ElasticPlastic* cm);
         
    ////////////////////////////////////////////////////////////////////////
    /*! \brief destructor  */
    ////////////////////////////////////////////////////////////////////////
    virtual ~ElasticPlastic();

    // clone
    ElasticPlastic* clone();
         
    ////////////////////////////////////////////////////////////////////////
    /*! \brief Put documentation here. */
    ////////////////////////////////////////////////////////////////////////
    virtual void addInitialComputesAndRequires(Task* task,
                                               const MPMMaterial* matl,
                                               const PatchSet* patches) const;

    ////////////////////////////////////////////////////////////////////////
    /*! \brief initialize  each particle's constitutive model data */
    ////////////////////////////////////////////////////////////////////////
    virtual void initializeCMData(const Patch* patch,
                                  const MPMMaterial* matl,
                                  DataWarehouse* new_dw);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief compute stable timestep for this patch */
    ////////////////////////////////////////////////////////////////////////
    virtual void computeStableTimestep(const Patch* patch,
                                       const MPMMaterial* matl,
                                       DataWarehouse* new_dw);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Put documentation here. */
    ////////////////////////////////////////////////////////////////////////
    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    ////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Compute stress at each particle in the patch 

      The plastic work is converted into a rate of temperature increase
      using an equation of the form
      \f[
         \dot{T} = \frac{\chi}{\rho C_p}(\sigma:D^p)
      \f]
    */
    ////////////////////////////////////////////////////////////////////////
    virtual void computeStressTensor(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Put documentation here. */
    ////////////////////////////////////////////////////////////////////////
    virtual void addComputesAndRequires(Task* ,
                                        const MPMMaterial* ,
                                        const PatchSet* ,
                                        const bool ) const;

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Compute Stress Tensor Implicit */
    ////////////////////////////////////////////////////////////////////////
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

    ////////////////////////////////////////////////////////////////////////
    /*! \brief carry forward CM data for RigidMPM */
    ////////////////////////////////////////////////////////////////////////
    virtual void carryForward(const PatchSubset* patches,
                              const MPMMaterial* matl,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Put documentation here. */
    ////////////////////////////////////////////////////////////////////////
    virtual void addRequiresDamageParameter(Task* task,
                                            const MPMMaterial* matl,
                                            const PatchSet* patches) const;


    ////////////////////////////////////////////////////////////////////////
    /*! \brief Put documentation here. */
    ////////////////////////////////////////////////////////////////////////
    virtual void getDamageParameter(const Patch* patch, 
                                    ParticleVariable<int>& damage, int dwi,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Put documentation here. */
    ////////////////////////////////////////////////////////////////////////
    virtual void allocateCMDataAddRequires(Task* task, const MPMMaterial* matl,
                                           const PatchSet* patch, 
                                           MPMLabel* lb) const;

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Put documentation here. */
    ////////////////////////////////////////////////////////////////////////
    virtual void allocateCMDataAdd(DataWarehouse* new_dw,
                                   ParticleSubset* subset,
                                   map<const VarLabel*, 
                                   ParticleVariableBase*>* newState,
                                   ParticleSubset* delset,
                                   DataWarehouse* old_dw);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Put documentation here. */
    ////////////////////////////////////////////////////////////////////////
    void scheduleCheckNeedAddMPMMaterial(Task* task,
                                         const MPMMaterial* matl,
                                         const PatchSet* patches) const;

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Put documentation here. */
    ////////////////////////////////////////////////////////////////////////
    virtual void checkNeedAddMPMMaterial(const PatchSubset* patches,
                                         const MPMMaterial* matl,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief initialize  each particle's constitutive model data */
    ////////////////////////////////////////////////////////////////////////
    virtual void addParticleState(std::vector<const VarLabel*>& from,
                                  std::vector<const VarLabel*>& to);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Sockets for MPM-ICE */
    ////////////////////////////////////////////////////////////////////////
    virtual double computeRhoMicroCM(double pressure,
                                     const double p_ref,
                                     const MPMMaterial* matl);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Sockets for MPM-ICE */
    ////////////////////////////////////////////////////////////////////////
    virtual void computePressEOSCM(double rho_m, double& press_eos,
                                   double p_ref,
                                   double& dp_drho, double& ss_new,
                                   const MPMMaterial* matl);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Sockets for MPM-ICE */
    ////////////////////////////////////////////////////////////////////////
    virtual double getCompressibility();

  protected:

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Compute Stilde, epdot, ep, and delGamma using 
               Simo's approach */
    ////////////////////////////////////////////////////////////////////////
    void computeStilde(const Matrix3& trialS,
                       const double& delT,
                       const MPMMaterial* matl,
                       const particleIndex idx,
                       Matrix3& Stilde,
                       PlasticityState* state,
                       double& delGamma);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Compute the quantity 
               \f$d(\gamma)/dt * \Delta T = \Delta \gamma \f$ 
               using Newton iterative root finder
        where \f$ d_p = \dot\gamma d(sigma_y)/d(sigma) \f$ */
    ////////////////////////////////////////////////////////////////////////
    double computeDeltaGamma(const double& delT,
                             const double& tolerance,
                             const double& normTrialS,
                             const MPMMaterial* matl,
                             const particleIndex idx,
                             PlasticityState* state);

    ////////////////////////////////////////////////////////////////////////
    /*! Compute the elastic tangent modulus tensor for isotropic
        materials
        Assume: [stress] = [s11 s22 s33 s23 s31 s12]
                [strain] = [e11 e22 e33 2e23 2e31 2e12] */
    ////////////////////////////////////////////////////////////////////////
    void computeElasticTangentModulus(const double& K,
                                      const double& mu,
                                      double Ce[6][6]);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Compute the elastic tangent modulus tensor for isotropic
      materials */
    ////////////////////////////////////////////////////////////////////////
    void computeElasticTangentModulus(double bulk,
                                      double shear,
                                      TangentModulusTensor& Ce);

    ////////////////////////////////////////////////////////////////////////
    /*! Compute the elastic-plastic tangent modulus tensor for isotropic
        materials for use in the implicit stress update
        Assume: [stress] = [s11 s22 s33 s23 s31 s12]
                [strain] = [e11 e22 e33 2e23 2e31 2e12] 
        Uses alogorithm for small strain plasticity (Simo 1998, p.124) */
    ////////////////////////////////////////////////////////////////////////
    void computeEPlasticTangentModulus(const double& K,
                                       const double& mu,
                                       const double& delGamma,
                                       const double& normTrialS,
                                       const particleIndex idx,
                                       const Matrix3& n,
                                       PlasticityState* state,
                                       double Cep[6][6],
                                       bool consistent);

    ////////////////////////////////////////////////////////////////////////
    /*! compute stress at each particle in the patch */
    ////////////////////////////////////////////////////////////////////////
    void computeStressTensorImplicit(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);

    ////////////////////////////////////////////////////////////////////////
    /*! Compute K matrix */
    ////////////////////////////////////////////////////////////////////////
    void computeStiffnessMatrix(const double B[6][24],
                                const double Bnl[3][24],
                                const double D[6][6],
                                const Matrix3& sig,
                                const double& vol_old,
                                const double& vol_new,
                                double Kmatrix[24][24]);

    ////////////////////////////////////////////////////////////////////////
    /*! Compute stiffness matrix for geomtric nonlinearity */
    ////////////////////////////////////////////////////////////////////////
    void BnlTSigBnl(const Matrix3& sig, const double Bnl[3][24],
                    double Kgeo[24][24]) const;

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Compute Porosity.
      
    The evolution of porosity is given by \n
    \f$
    \dot{f} = \dot{f}_{nucl} + \dot{f}_{grow}
    \f$ \n
    where
    \f$
    \dot{f}_{grow} = (1-f) D^p_{kk}
    \f$ \n
    \f$ D^p_{kk} = Tr(D^p) \f$, and \f$ D^p \f$ is the rate of plastic
    deformation, and, \n
    \f$
    \dot{f}_{nucl} = A \dot{\epsilon}^p
    \f$  \n
    with 
    \f$
    A = f_n/(s_n \sqrt{2\pi}) \exp [-1/2 (\epsilon^p - \epsilon_n)^2/s_n^2]
    \f$\n
    \f$ f_n \f$ is the volume fraction of void nucleating particles , 
    \f$ \epsilon_n \f$ is the mean of the normal distribution of nucleation
    strains, and \f$ s_n \f$ is the standard deviation of the distribution.
   
    References:
    1) Ramaswamy, S. and Aravas, N., 1998, Comput. Methods Appl. Mech. Engrg.,
    163, 33-53.
    2) Bernauer, G. and Brocks, W., 2002, Fatigue Fract. Engng. Mater. Struct.,
    25, 363-384.
    */
    ////////////////////////////////////////////////////////////////////////
    double updatePorosity(const Matrix3& rateOfDeform,
                          double delT, double oldPorosity,
                          double plasticStrain);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Calculate void nucleation factor */
    ////////////////////////////////////////////////////////////////////////
    inline double voidNucleationFactor(double plasticStrain);

  protected:

    void initializeLocalMPMLabels();

    void getInitialPorosityData(ProblemSpecP& ps);

    void getInitialDamageData(ProblemSpecP& ps);

    void getSpecificHeatData(ProblemSpecP& ps);

    void setErosionAlgorithm();

    double computeSpecificHeat(double T);

  };

} // End namespace Uintah

#endif  // __ELASTIC_PLASTIC_H__ 
