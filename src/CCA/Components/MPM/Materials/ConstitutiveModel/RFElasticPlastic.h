/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __RFELASTICPLASTIC_H__
#define __RFELASTICPLASTIC_H__


#include "ConstitutiveModel.h"
#include "PlasticityModels/YieldCondition.h"
#include "PlasticityModels/StabilityCheck.h"
#include "PlasticityModels/FlowModel.h"
#include "PlasticityModels/MPMEquationOfState.h"
#include "PlasticityModels/ShearModulusModel.h"
#include "PlasticityModels/MeltingTempModel.h"
#include "PlasticityModels/SpecificHeatModel.h"
#include "PlasticityModels/DevStressModel.h"

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

    The rate of deformation and stress is rotated to material configuration 
    before the updated values are calculated.  The left stretch and rotation 
    are updated incrementatlly to get the deformation gradient.

    Needs :
    1) Isotropic elastic moduli.
    2) Flow rule in the form of a Plasticity Model.
    3) Yield condition.
    4) Stability condition.
    6) Shear modulus model.
    7) Melting temperature model.
    8) Specific heat model.

    \Modified by Jim Guilkey to use energy based EOS

    \warning Only isotropic materials, von-Mises type yield conditions, 
    associated flow rule, high strain rate.
  */
  /////////////////////////////////////////////////////////////////////////////

class RFElasticPlastic : public ConstitutiveModel {

  public:
    // Create datatype for storing model parameters
    struct CMData {
      double Bulk;    /*< Bulk modulus */
      double Shear;   /*< Shear Modulus */
      double alpha;   /*< Coeff. of thermal expansion */
      double Chi;     /*< Taylor-Quinney coefficient */
      double sigma_crit; /*< Critical stress */
      //********** Concentration Component****************************
      double vol_exp_coeff; //Volume expansion coefficient
      //********** Concentration Component****************************
    };   

    const VarLabel* pPlasticStrainLabel;  
    const VarLabel* pPlasticStrainRateLabel;  
    const VarLabel* pEnergyLabel;  

    const VarLabel* pPlasticStrainLabel_preReloc;  
    const VarLabel* pPlasticStrainRateLabel_preReloc;  
    const VarLabel* pEnergyLabel_preReloc;

  protected:

    CMData           d_initialData;

    double d_tol;
    double d_initialMaterialTemperature;
    double d_isothermal;
    double d_partial_vol;
    bool   d_doIsothermal;
    bool   d_useModifiedEOS;
    bool   d_computeSpecificHeat;
    bool   d_checkTeplaFailureCriterion;
    bool   d_doMelting;
    bool   d_checkStressTriax;

    std::string  d_plasticConvergenceAlgo;

    YieldCondition*     d_yield;
    StabilityCheck*     d_stable;
    FlowModel*          d_flow;
    MPMEquationOfState* d_eos;
    ShearModulusModel*  d_shear;
    MeltingTempModel*   d_melt;
    SpecificHeatModel*  d_Cp;
    DevStressModel*     d_devStress;
         
  private:
    // Prevent copying of this class
    // copy constructor
    RFElasticPlastic& operator=(const RFElasticPlastic &cm);

  public:

    ////////////////////////////////////////////////////////////////////////
    /*! \brief constructors */
    ////////////////////////////////////////////////////////////////////////
    RFElasticPlastic(ProblemSpecP& ps,MPMFlags* flag);
         
    ////////////////////////////////////////////////////////////////////////
    /*! \brief destructor  */
    ////////////////////////////////////////////////////////////////////////
    virtual ~RFElasticPlastic();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone
    RFElasticPlastic* clone();
         
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
    virtual void computeStableTimeStep(const Patch* patch,
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
    /*! \brief carry forward CM data for RigidMPM */
    ////////////////////////////////////////////////////////////////////////
    virtual void carryForward(const PatchSubset* patches,
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

    virtual void addSplitParticlesComputesAndRequires(Task* task,
                                                      const MPMMaterial* matl,
                                                      const PatchSet* patches);

    virtual void splitCMSpecificParticleData(const Patch* patch,
                                             const int dwi,
                                             const int fourOrEight,
                                             ParticleVariable<int> &prefOld,
                                             ParticleVariable<int> &prefNew,
                                             const unsigned int oldNumPar,
                                             const unsigned int numNewPartNeeded,
                                             DataWarehouse* old_dw,
                                             DataWarehouse* new_dw);

  protected:
  
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

  protected:

    void initializeLocalMPMLabels();
  };

} // End namespace Uintah

#endif  // __RFELASTICPLASTIC_H__ 
