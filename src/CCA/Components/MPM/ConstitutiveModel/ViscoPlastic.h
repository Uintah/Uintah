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


#ifndef __VISCO_PLASTIC_H__
#define __VISCO_PLASTIC_H__


#include <CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include "ImplicitCM.h"
#include "PlasticityModels/YieldCondition.h"
#include "PlasticityModels/StabilityCheck.h"
#include "PlasticityModels/ViscoPlasticityModel.h"
/*#include "PlasticityModels/DamageModel.h"*/
#include "PlasticityModels/MPMEquationOfState.h"
#include <math.h>
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
    \class ViscoPlastic
    \brief Unified Viscoplastic model for ice - SUVIC-I
    \author Jonah Lee \n
    Department of Mechanical Engineering \n
    University of Alaska Fairbanks \n
    Copyright (C) 2008 University of Alaska Fairbanks

    Borrowed from ElasticPlastic.h

    The rate of deformation and stress is rotated to material configuration 
    before the updated values are calculated.  The left stretch and rotation 
    are updated incrementatlly to get the deformation gradient.

    Yield stress, back stress, drag stress are the main state variables.

    Needs :
    1) Isotropic elastic moduli.
    2) Flow rule in the form of a Plasticity Model.
    3) Yield condition.
    4) Stability condition.
    5) Damage model - after CNHDamage

    \warning Only SUVIC-I implemented. TODO-distill later for more general
    viscoplastic models; add newton-raphson local iteration
  */
  /////////////////////////////////////////////////////////////////////////////

  class ViscoPlastic : public ConstitutiveModel, public ImplicitCM {

  public:
    // Create datatype for storing basic (elastic mainly) model parameters
    struct CMData {
      double Bulk;    /*< Bulk modulus */
      double Shear;   /*< Shear Modulus */
      double alpha;   /*< Coeff. of thermal expansion */
    };  

    // Create datatype for failure criteria (only stress/strain now) 
    struct FailureVariableData {
      double mean;      /*< Mean failure variable */
      double std;       /*< Standard deviation of failure variable */
      std::string dist; /*< Failure variable distrinution */
      bool failureByStress; /*<Failure by maximum principle stress (default) */
      bool failureByPressure; /*<Failure by maximum tensile mean stress */
    };


    // Create datatype for storing state variable parameters
 
    const VarLabel* pLeftStretchLabel;  // For ViscoPlasticity
    const VarLabel* pRotationLabel;  // For ViscoPlasticity
    const VarLabel* pStrainRateLabel;  
    const VarLabel* pPlasticStrainLabel;  
    const VarLabel* pPlasticTempLabel;  
    const VarLabel* pPlasticTempIncLabel;  
    const VarLabel* pLocalizedLabel;  

    const VarLabel* pFailureVariableLabel; //For failure criteria
    const VarLabel* pFailureVariableLabel_preReloc;

    const VarLabel* pLeftStretchLabel_preReloc;  // For ViscoPlasticity
    const VarLabel* pRotationLabel_preReloc;  // For ViscoPlasticity
    const VarLabel* pStrainRateLabel_preReloc;  
    const VarLabel* pPlasticStrainLabel_preReloc;   
    const VarLabel* pPlasticTempLabel_preReloc;  
    const VarLabel* pPlasticTempIncLabel_preReloc;  
    const VarLabel* pLocalizedLabel_preReloc;  

  protected:

    CMData           d_initialData;

    FailureVariableData d_varf;

    double d_tol;
    double d_initialMaterialTemperature;
    bool   d_useModifiedEOS;

    bool   d_checkFailure;
    bool   d_removeParticles;
    bool   d_setStressToZero;
    bool   d_allowNoTension;
    bool   d_usePolarDecompositionRMB; /*< use RMB's polar decomposition */

    YieldCondition*     d_yield;
    StabilityCheck*     d_stable;
    ViscoPlasticityModel*    d_plastic;
    MPMEquationOfState* d_eos;
         
  private:

    // Prevent copying of this class
    // copy constructor
    //ViscoPlastic(const ViscoPlastic &cm);
    ViscoPlastic& operator=(const ViscoPlastic &cm);

    void getFailureVariableData(ProblemSpecP& ps);

    void setFailureVariableData(const ViscoPlastic* cm);

  public:

    ////////////////////////////////////////////////////////////////////////
    /*! \brief constructors */
    ////////////////////////////////////////////////////////////////////////
    ViscoPlastic(ProblemSpecP& ps,MPMFlags* flag);
    ViscoPlastic(const ViscoPlastic* cm);
         
    ////////////////////////////////////////////////////////////////////////
    /*! \brief destructor  */
    ////////////////////////////////////////////////////////////////////////
    virtual ~ViscoPlastic();
    
    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone
    ViscoPlastic* clone();
         
    ////////////////////////////////////////////////////////////////////////
    /*! \brief Initial CR */
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
    /*! \brief Computes and requires explicit. */
    ////////////////////////////////////////////////////////////////////////
    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches) const;

    ////////////////////////////////////////////////////////////////////////
    /*! 
      \brief Compute stress at each particle in the patch (explicit)

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
    /*! \brief Computes and Requires Implicit */
    ////////////////////////////////////////////////////////////////////////
    virtual void addComputesAndRequires(Task* task,
                                        const MPMMaterial* matl,
                                        const PatchSet* patches,
                                        const bool recursion,
                                        const bool SchedParent) const;

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Compute Stress Tensor Implicit */
    ////////////////////////////////////////////////////////////////////////
    virtual void computeStressTensor(const PatchSubset* patches,
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
    /*! \brief Get the increment in plastic temperature. */
    ////////////////////////////////////////////////////////////////////////
    void getPlasticTemperatureIncrement(ParticleSubset* pset,
                                        DataWarehouse* new_dw,
                                        ParticleVariable<double>& T) ;

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

    // Modify the stress if particle has failed
    bool updateFailedParticlesAndModifyStress(const Matrix3& bb,
                                              const double& pFailureVariable,
                                              const int& pLocalized,
                                              int& pLocalized_new,
                                              Matrix3& pStress_new,
                                              const long64 particleID,
                                              const double temp_new,
                                              const double Tm_cur);


    ////////////////////////////////////////////////////////////////////////
    /*! \brief Compute the updated left stretch and rotation tensors */
    ////////////////////////////////////////////////////////////////////////
    void computeUpdatedVR(const double& delT,
                          const Matrix3& DD, 
                          const Matrix3& WW,
                          Matrix3& VV, 
                          Matrix3& RR);  

    ////////////////////////////////////////////////////////////////////////
    /*! \brief Compute the rate of rotation tensor */
    ////////////////////////////////////////////////////////////////////////
    Matrix3 computeRateofRotation(const Matrix3& tensorV, 
                                  const Matrix3& tensorD,
                                  const Matrix3& tensorW);

    ////////////////////////////////////////////////////////////////////////
    /*! compute stress at each particle in the patch */
    ////////////////////////////////////////////////////////////////////////
    void computeStressTensorImplicit(const PatchSubset* patches,
                                     const MPMMaterial* matl,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw);


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
    virtual void computeEPlasticTangentModulus(const double& K,
                                       const double& mu,
                                       const double& delGamma,
                                       const double& normTrialS,
                                       const particleIndex idx,
                                       const Matrix3& n,
                                       PlasticityState* state,
                                       double Cep[6][6]);

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

    // Convert to double [6][6] (Voigt form)
    void convertToVoigtForm(const TangentModulusTensor Ce, double D[6][6]);


  private:

    void initializeLocalMPMLabels();

  };

} // End namespace Uintah

#endif  // __VISCO_PLASTIC_H__ 
