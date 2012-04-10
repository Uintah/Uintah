/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef __ELASTIC_PLASTICHP_H__
#define __ELASTIC_PLASTICHP_H__


#include "ConstitutiveModel.h"
#include "ImplicitCM.h"
#include "PlasticityModels/YieldCondition.h"
#include "PlasticityModels/StabilityCheck.h"
#include "PlasticityModels/FlowModel.h"
#include "PlasticityModels/DamageModel.h"
#include "PlasticityModels/MPMEquationOfState.h"
#include "PlasticityModels/ShearModulusModel.h"
#include "PlasticityModels/MeltingTempModel.h"
#include "PlasticityModels/SpecificHeatModel.h"
#include "PlasticityModels/DeviatoricStressModel.h"
#include <cmath>
#include <Core/Math/Matrix3.h>
#include <Core/Math/TangentModulusTensor.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/Variables/NCVariable.h>

namespace Uintah {

  class MPMLabel;
  class MPMFlags;

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class ElasticPlasticHP
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
    6) Shear modulus model.
    7) Melting temperature model.
    8) Specific heat model.

    \Modified by Jim Guilkey to use energy based EOS

    \warning Only isotropic materials, von-Mises type yield conditions, 
    associated flow rule, high strain rate.
  */
  /////////////////////////////////////////////////////////////////////////////

  class ElasticPlasticHP : public ConstitutiveModel, public ImplicitCM {

  public:
    // Create datatype for storing model parameters
    struct CMData {
      double Bulk;    /*< Bulk modulus */
      double Shear;   /*< Shear Modulus */
      double alpha;   /*< Coeff. of thermal expansion */
      double Chi;     /*< Taylor-Quinney coefficient */
      double sigma_crit; /*< Critical stress */
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
    //struct CpData {
    //  double A;
    //  double B;
    //  double C;
    //  double n;
    //};

    const VarLabel* pRotationLabel;  // For Hypoelastic-plasticity
    const VarLabel* pStrainRateLabel;  
    const VarLabel* pPlasticStrainLabel;  
    const VarLabel* pPlasticStrainRateLabel;  
    const VarLabel* pDamageLabel;  
    const VarLabel* pPorosityLabel;  
    const VarLabel* pLocalizedLabel;  
    const VarLabel* pEnergyLabel;  

    const VarLabel* pRotationLabel_preReloc;  // For Hypoelastic-plasticity
    const VarLabel* pStrainRateLabel_preReloc;  
    const VarLabel* pPlasticStrainLabel_preReloc;  
    const VarLabel* pPlasticStrainRateLabel_preReloc;  
    const VarLabel* pDamageLabel_preReloc;  
    const VarLabel* pPorosityLabel_preReloc;  
    const VarLabel* pLocalizedLabel_preReloc;  
    const VarLabel* pEnergyLabel_preReloc;

  protected:

    CMData           d_initialData;
    PorosityData     d_porosity;
    ScalarDamageData d_scalarDam;
    //CpData           d_Cp;
    
    double d_tol;
    double d_initialMaterialTemperature;
    double d_isothermal;
    bool   d_doIsothermal;
    bool   d_useModifiedEOS;
    bool   d_evolvePorosity;
    bool   d_evolveDamage;
    bool   d_computeSpecificHeat;
    bool   d_checkTeplaFailureCriterion;
    bool   d_doMelting;
    bool   d_checkStressTriax;

    std::string  d_plasticConvergenceAlgo;
    // Erosion algorithms
    bool   d_setStressToZero;
    bool   d_allowNoTension;
    bool   d_allowNoShear;

    YieldCondition*     d_yield;
    StabilityCheck*     d_stable;
    FlowModel*          d_flow;
    DamageModel*        d_damage;
    MPMEquationOfState* d_eos;
    ShearModulusModel*  d_shear;
    MeltingTempModel*   d_melt;
    SpecificHeatModel*  d_Cp;
    DeviatoricStressModel*  d_devStress;
         
  private:
    // Prevent copying of this class
    // copy constructor
    ElasticPlasticHP& operator=(const ElasticPlasticHP &cm);

  public:

    ////////////////////////////////////////////////////////////////////////
    /*! \brief constructors */
    ////////////////////////////////////////////////////////////////////////
    ElasticPlasticHP(ProblemSpecP& ps,MPMFlags* flag);
    ElasticPlasticHP(const ElasticPlasticHP* cm);
         
    ////////////////////////////////////////////////////////////////////////
    /*! \brief destructor  */
    ////////////////////////////////////////////////////////////////////////
    virtual ~ElasticPlasticHP();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone
    ElasticPlasticHP* clone();
         
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
                                        const bool ,
                                        const bool ) const;

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Compute Stress Tensor Implicit */
    ////////////////////////////////////////////////////////////////////////
    virtual void computeStressTensorImplicit(const PatchSubset* patches,
                                             const MPMMaterial* matl,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw,
                                             Solver* solver,
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
                                     const MPMMaterial* matl, 
                                     double temperature,
                                     double rho_guess);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Sockets for MPM-ICE */
    ////////////////////////////////////////////////////////////////////////
    virtual void computePressEOSCM(double rho_m, double& press_eos,
                                   double p_ref,
                                   double& dp_drho, double& ss_new,
                                   const MPMMaterial* matl, 
                                   double temperature);

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Sockets for MPM-ICE */
    ////////////////////////////////////////////////////////////////////////
    virtual double getCompressibility();

  protected:
  
    ////////////////////////////////////////////////////////////////////////
    /*! \brief Compute Plastic State using Biswajit's approach */
    ////////////////////////////////////////////////////////////////////////  
    bool computePlasticStateBiswajit( PlasticityState* state, 
                                      constParticleVariable<double>& pPlasticStrain,
                                      constParticleVariable<double>& pStrainRate,
                                      const Matrix3& sigma,
                                      const Matrix3& trialS,
                                      const Matrix3& tensorEta,
                                      Matrix3& tensorS,
                                      double& delGamma,
                                      double& flowStress,
                                      double& porosity,
                                      double& mu_cur,
                                      const double delT,
                                      const MPMMaterial* matl,
                                      const int idx);
    
    ////////////////////////////////////////////////////////////////////////
    /*! \brief Compute Stilde, epdot, ep, and delGamma using 
               Simo's approach */
    ////////////////////////////////////////////////////////////////////////
    void computePlasticStateViaRadialReturn(const Matrix3& trialS,
                                            const double& delT,
                                            const MPMMaterial* matl,
                                            const particleIndex idx,
                                            PlasticityState* state,
                                            Matrix3& nn,
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
                                       const Matrix3& trialStess,
                                       const particleIndex idx,
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

    void setErosionAlgorithm();

    //void getSpecificHeatData(ProblemSpecP& ps);
    //double computeSpecificHeat(double T);

  };

} // End namespace Uintah

#endif  // __ELASTIC_PLASTICHP_H__ 
