/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifndef __GAOELASTIC_H__
#define __GAOELASTIC_H__


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
  class ReactionDiffusionLabel;

  //*********************************************************
	// Reactive Flow is a major refactor of the below model
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
    5) Damage model.
    6) Shear modulus model.
    7) Melting temperature model.
    8) Specific heat model.

    \Modified by Jim Guilkey to use energy based EOS

    \warning Only isotropic materials, von-Mises type yield conditions, 
    associated flow rule, high strain rate.
  */
  /////////////////////////////////////////////////////////////////////////////

  class GaoElastic : public ConstitutiveModel, public ImplicitCM {

  public:
    // Create datatype for storing model parameters
    struct CMData {
      double Bulk;    /*< Bulk modulus */
      double Shear;   /*< Shear Modulus */
			double vol_exp_coeff; /* Volume Expansion Coeff for concentration */
    };   

    const VarLabel* pRotationLabel;  // For Hypoelastic-plasticity
    const VarLabel* pStrainRateLabel;  
    const VarLabel* pLocalizedLabel;  
    const VarLabel* pEnergyLabel;  

    const VarLabel* pRotationLabel_preReloc;  // For Hypoelastic-plasticity
    const VarLabel* pStrainRateLabel_preReloc;  
    const VarLabel* pLocalizedLabel_preReloc;  
    const VarLabel* pEnergyLabel_preReloc;

  protected:

    CMData           d_initialData;
    
    double d_tol;
    double d_initialMaterialTemperature;
    double d_isothermal;
    double alpha;
    bool   d_doIsothermal;
    bool   d_useModifiedEOS;
    bool   d_evolvePorosity;
    bool   d_evolveDamage;
    bool   d_computeSpecificHeat;
    bool   d_checkTeplaFailureCriterion;
    bool   d_doMelting;
    bool   d_checkStressTriax;

    std::string  d_plasticConvergenceAlgo;
         
  private:
    // Prevent copying of this class
    // copy constructor
    GaoElastic& operator=(const GaoElastic &cm);
    ReactionDiffusionLabel* d_rdlb;

  public:

    ////////////////////////////////////////////////////////////////////////
    /*! \brief constructors */
    ////////////////////////////////////////////////////////////////////////
    GaoElastic(ProblemSpecP& ps,MPMFlags* flag);
    GaoElastic(const GaoElastic* cm);
         
    ////////////////////////////////////////////////////////////////////////
    /*! \brief destructor  */
    ////////////////////////////////////////////////////////////////////////
    virtual ~GaoElastic();

    virtual void outputProblemSpec(ProblemSpecP& ps,bool output_cm_tag = true);

    // clone
    GaoElastic* clone();
         
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
    /*! \brief Put documentation here. */
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

    void initializeLocalMPMLabels();

  };

} // End namespace Uintah

#endif  // __GAOELASTIC_H__ 
