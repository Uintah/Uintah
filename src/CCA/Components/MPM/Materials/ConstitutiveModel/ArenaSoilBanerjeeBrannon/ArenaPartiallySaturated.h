/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 * Copyright (c) 2013-2014 Callaghan Innovation, New Zealand
 * Copyright (c) 2015-2017 Parresia Research Limited, New Zealand
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



#ifndef __ARENA_PARTIALLY_SATURATED__
#define __ARENA_PARTIALLY_SATURATED__


#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ArenaSoilBanerjeeBrannon/Models/ModelState_Arena.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ArenaSoilBanerjeeBrannon/Models/ElasticModuliModel.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ArenaSoilBanerjeeBrannon/Models/YieldCondition.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ArenaSoilBanerjeeBrannon/Models/Pressure_Air.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ArenaSoilBanerjeeBrannon/Models/Pressure_Water.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ArenaSoilBanerjeeBrannon/Models/Pressure_Granite.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/DataWarehouseP.h>

#include <cmath>

namespace Uintah {
  class MPMLabel;
  class MPMFlags;
  class Matrix3;
  class VarLabel;
  class Patch;
  class Task;
  class MPMMaterial;
}

namespace Uintah {
  typedef std::map<const VarLabel*, ParticleVariableBase*> ParticleLabelVariableMap;
}

namespace Vaango {

  class ArenaPartiallySaturated : public Uintah::ConstitutiveModel {

  public:
    static const double one_third;
    static const double two_third;
    static const double four_third;
    static const double sqrt_two;
    static const double one_sqrt_two;
    static const double sqrt_three;
    static const double one_sqrt_three;
    static const double one_sixth;
    static const double one_ninth;
    static const double pi;
    static const double pi_fourth;
    static const double pi_half;
    static const Uintah::Matrix3 Identity;
    static const Uintah::Matrix3 Zero;

    // Create datatype for storing model parameters
    struct CMData {
      double yield_scale_fac;
      double consistency_bisection_tolerance;
      double max_bisection_iterations;
      double subcycling_characteristic_number;
      bool   use_disaggregation_algorithm;
      bool   do_damage;
      double K0_Murnaghan_EOS;
      double n_Murnaghan_EOS;
    };

    // Initial porosity and saturation parameters
    struct FluidEffectParameters {
      double phi_ref;  // reference porosity used during parameter calibration
      double phi0;     // initial porosity
      double Sw0;      // initial water saturation
      double pbar_w0;  // initial fluid pressure
    };

    // Crush Curve Model parameters
    struct CrushParameters {
      double p0;
      double p1;
      double p1_sat;
      double p1_density_scale_fac;
      double p2;
      double p3;
    };

    // Damage Model parameters
    struct DamageParameters {
      double fSpeed;               // Failure speed
      double tFail;                // Time at failure
      double ep_f_eq;              // Equivalent plastic strain at failure
    };

    // For initialization with body force
    bool d_initializeWithBodyForce;
    Uintah::Point d_surfaceRefPoint;

    const Uintah::VarLabel* pStressQSLabel;                      // Quasistatic stress
    const Uintah::VarLabel* pStressQSLabel_preReloc;
    const Uintah::VarLabel* pElasticVolStrainLabel;              // Elastic Volumetric Strain
    const Uintah::VarLabel* pElasticVolStrainLabel_preReloc;

    // Internal variables
    const Uintah::VarLabel* pPlasticStrainLabel;                 // Plastic Strain
    const Uintah::VarLabel* pPlasticStrainLabel_preReloc;
    const Uintah::VarLabel* pPlasticCumEqStrainLabel;            // Equivalent plastic strain
    const Uintah::VarLabel* pPlasticCumEqStrainLabel_preReloc;   // (monotonically increasing)
    const Uintah::VarLabel* pPlasticVolStrainLabel;              // Plastic Volumetric Strain
    const Uintah::VarLabel* pPlasticVolStrainLabel_preReloc;
    const Uintah::VarLabel* pBackstressLabel;                    // Pore pressure
    const Uintah::VarLabel* pBackstressLabel_preReloc; 
    const Uintah::VarLabel* pCapXLabel;                          // Hydrostatic strength
    const Uintah::VarLabel* pCapXLabel_preReloc; 
    const Uintah::VarLabel* pPorosityLabel;                      // Porosity
    const Uintah::VarLabel* pPorosityLabel_preReloc; 
    const Uintah::VarLabel* pSaturationLabel;                    // Saturation
    const Uintah::VarLabel* pSaturationLabel_preReloc; 

    // Disaggregation, failure, and damage
    const Uintah::VarLabel* pP3Label;                            // Evolution of parameter P3
    const Uintah::VarLabel* pP3Label_preReloc;
    const Uintah::VarLabel* pCoherenceLabel;                     // Coherence parameter
    const Uintah::VarLabel* pCoherenceLabel_preReloc;
    const Uintah::VarLabel* pTGrowLabel;                         // t_grow parameter
    const Uintah::VarLabel* pTGrowLabel_preReloc;

  private:

    ElasticModuliModel*      d_elastic;
    YieldCondition*          d_yield;

    /* Tangent bulk modulus models for air, water, granite */
    Pressure_Air     d_air;
    Pressure_Water   d_water;
    Pressure_Granite d_solid;

    CMData                d_cm;
    FluidEffectParameters d_fluidParam;
    CrushParameters       d_crushParam;
    DamageParameters      d_damageParam;

    // Scaling factors
    double d_modulus_scale_fac;   // density based modulus scaling
    double d_strength_scale_fac;  // density based strength scaling

    // Prevent copying of this class
    // copy constructor
    ArenaPartiallySaturated& operator=(const ArenaPartiallySaturated &cm);

    void initializeLocalMPMLabels();
    void checkInputParameters();

  public:
    // constructor
    ArenaPartiallySaturated(Uintah::ProblemSpecP& ps, Uintah::MPMFlags* flag);
    ArenaPartiallySaturated(const ArenaPartiallySaturated* cm);

    // destructor
    virtual ~ArenaPartiallySaturated();

    virtual void outputProblemSpec(Uintah::ProblemSpecP& ps,bool output_cm_tag = true);

    // clone
    ArenaPartiallySaturated* clone();

    /*! Get parameters */
    ParameterDict getParameters() const {
      ParameterDict params;
      params["phi0"]    = d_fluidParam.phi0;
      params["Sw0"]     = d_fluidParam.Sw0;
      params["pbar_w0"] = d_fluidParam.pbar_w0;
      params["p0"]      = d_crushParam.p0;
      params["p1"]      = d_crushParam.p1;
      params["p1_sat"]  = d_crushParam.p1_sat;
      params["p2"]      = d_crushParam.p2;
      params["p3"]      = d_crushParam.p3;
      return params;
    }

    // compute stable timestep for this patch
    virtual void computeStableTimeStep(const Uintah::Patch* patch,
                                       const Uintah::MPMMaterial* matl,
                                       Uintah::DataWarehouse* new_dw);

    // compute stress at each particle in the patch
    virtual void computeStressTensor(const Uintah::PatchSubset* patches,
                                     const Uintah::MPMMaterial* matl,
                                     Uintah::DataWarehouse* old_dw,
                                     Uintah::DataWarehouse* new_dw);

  private: //non-uintah mpm constitutive specific functions

    void initializeInternalVariables(const Uintah::Patch* patch,
                                     const Uintah::MPMMaterial* matl,
                                     Uintah::ParticleSubset* pset,
                                     Uintah::DataWarehouse* new_dw,
                                     ParameterDict& params);

    //////////////////////////////////////////////////////////////////////////
    /**
     * Function: 
     *   rateIndependentPlasticUpdate
     *
     * Purpose:
     *   Rate-independent plasticity:
     *   Divides the strain increment into substeps, and calls substep function
     *   to copute the stress and internal variables
     *
     * Inputs:
     *   D          = rate of deformation
     *   delT       = time increment
     *   idx        = Particle index
     *   particleID = long64 ID
     *   state_old  = state at t = t_n
     *
     * Outputs:
     *
     *   state_new  = state at t = t_(n+1)
     *
     * Returns: True for success; False for failure
     */
    //////////////////////////////////////////////////////////////////////////
    bool rateIndependentPlasticUpdate(const Uintah::Matrix3& D,
                                      const double& delT, 
                                      Uintah::particleIndex idx, 
                                      Uintah::long64 pParticleID, 
                                      const ModelState_Arena& state_old,
                                      ModelState_Arena& state_new);

    //////////////////////////////////////////////////////////////////////////
    /** 
     * Function: rateDependentPlasticUpdate
     *
     * Purpose:
     *   Rate-dependent plastic step
     *   Compute the new dynamic stress from the old dynamic stress and the new and old QS stress
     *   using Duvaut-Lions rate dependence, as described in "Elements of Phenomenological Plasticity",
     *   by RM Brannon.
     *
     * Inputs:
     *   D                = rate of deformation
     *   delT             = time increment
     *   stateStatic_old  = Quasistatic stress state at t = t_n
     *   stateStatic_new  = Quasistatic stress state at t = t_n+1
     *   stateDynamic_old = Dynamic stress state at t = t_n
     *
     * Outputs:
     *
     *   pStress_new  = stress state at t = t_(n+1)
     *
     * Returns: True for rate-dependent plasticity; False for rate-independent plasticity
     */
    bool rateDependentPlasticUpdate(const Uintah::Matrix3& D,
                                    const double& delT,
                                    const ModelState_Arena& stateStatic_old,
                                    const ModelState_Arena& stateStatic_new,
                                    const ModelState_Arena& stateDynamic_old,
                                    Uintah::Matrix3& pStress_new);

    //////////////////////////////////////////////////////////////////////////
    /** 
     * Method: computeElasticProperties
     *
     * Purpose: 
     *   Compute the bulk and shear mdoulus at a given state
     *
     * Inputs:
     *   state = state
     * 
     * Modifies
     *   state.bulkModulus
     *   state.shearModulus
     */
    //////////////////////////////////////////////////////////////////////////
    void computeElasticProperties(ModelState_Arena& state);

    //////////////////////////////////////////////////////////////////////////
    /** 
     * Method: computeTrialStress
     *
     * Purpose: 
     *   Compute the trial stress for some increment in strain assuming linear elasticity
     *   over the step.
     *
     * Inputs:
     *   state_old = state at t = t_n 
     *   strain_inc = strain increment (D * delT)
     *     D = rate of deformation
     *     delT = time step 
     * 
     * Returns
     *   stress_trial
     */
    //////////////////////////////////////////////////////////////////////////
    Uintah::Matrix3 computeTrialStress(const ModelState_Arena& state_old,
                                       const Uintah::Matrix3& strain_inc);

    //////////////////////////////////////////////////////////////////////////
    /** 
     * Method: computeStepDivisions
     *
     * Purpose: 
     *   Compute the number of step divisions (substeps) based on a comparison
     *   of the trial stress relative to the size of the yield surface, as well
     *   as change in elastic properties between sigma_n and sigma_trial.
     *
     * Inputs:
     *   idx            = particle index
     *   particleID     = long64 ID
     *   state_substep  = state at the current subsetp 
     *   state_trial    = the trial state
     * 
     * Returns
     *   nSub = the number of substeps
     */
    //////////////////////////////////////////////////////////////////////////
    int computeStepDivisions(Uintah::particleIndex idx,
                             Uintah::long64 particleID,
                             const ModelState_Arena& state_substep,
                             const ModelState_Arena& state_trial);

    //////////////////////////////////////////////////////////////////////////
    /** 
     * Method: computeSubstep
     *
     * Purpose: 
     *   Computes the updated stress state for a substep that may be either 
     *   elastic, plastic, or partially elastic.   
     *
     * Inputs:
     *   D              = rate of deformation tensor
     *   dt             = substep time increment
     *   state_old      = state at start of substep
     * 
     * Outputs:
     *   state_new      = updated state at end of substep
     *
     * Returns:
     *   True for success; false for failure
     */
    //////////////////////////////////////////////////////////////////////////
    bool computeSubstep(const Uintah::Matrix3& D,
                        const double& dt,
                        const ModelState_Arena& state_old,
                        ModelState_Arena& state_new);

    //////////////////////////////////////////////////////////////////////////
    /** 
     * Method: nonHardeningReturn
     *
     * Purpose: 
     *   Computes a non-hardening return to the yield surface in the meridional profile
     *   (constant Lode angle) based on the current values of the internal state variables
     *   and elastic properties.  Returns the updated stress and  the increment in plastic
     *   strain corresponding to this return.
     *
     *   NOTE: all values of r and z in this function are transformed!
     *
     * Inputs:
     *   strain_inc     = total strain icremenet = D*dt
     *     D              = rate of deformation tensor
     *     dt             = substep time increment
     *   state_old      = state at start of substep
     *   state_trial    = trial state at start of substep
     *   params         = yield condition parameters
     * 
     * Outputs:
     *   sig_new                 = updated stress at end of substep
     *   plasticStrain_inc_new   = updated plastic strain incremente at end of substep
     *
     * Returns:
     *   true  = success
     *   false = failure
     */
    //////////////////////////////////////////////////////////////////////////
    bool nonHardeningReturn(const Uintah::Matrix3& strain_inc,
                            const ModelState_Arena& state_old,
                            const ModelState_Arena& state_trial,
                            Uintah::Matrix3& sig_new,
                            Uintah::Matrix3& plasticStrain_inc_new);

    //////////////////////////////////////////////////////////////////////////
    /**
     * Method: consistencyBisection
     *
     * Purpose: 
     *   Find the updated stress for hardening plasticity using the consistency bisection 
     *   algorithm
     *   Returns whether the procedure is sucessful orhas failed
     *
     * Inputs:
     *   deltaEps_new = strain increment for the substep
     *   state_old    = state at the beginning of the substep 
     *   state_trial  = trial state
     *   deltaEps_p_0 = plastic strain increment at the beginning of substep
     *   sig_0        = stress at the beginning of substep
     *   params       = yield condition parameters
     *
     * Outputs:
     *   state_old    = state at the end of the substep 
     *
     * Returns:
     *   isSuccess    = true if success, else false
     */
    //////////////////////////////////////////////////////////////////////////
    bool consistencyBisectionSimplified(const Uintah::Matrix3& deltaEps_new,
                                        const ModelState_Arena& state_old, 
                                        const ModelState_Arena& state_trial,
                                        const Uintah::Matrix3& deltaEps_p_0, 
                                        const Uintah::Matrix3& sig_0, 
                                        ModelState_Arena& state_new);
    bool consistencyBisection(const Uintah::Matrix3& deltaEps_new,
                              const ModelState_Arena& state_old, 
                              const ModelState_Arena& state_trial,
                              const Uintah::Matrix3& deltaEps_p_0, 
                              const Uintah::Matrix3& sig_0, 
                              ModelState_Arena& state_new);

    //////////////////////////////////////////////////////////////////////////
    /** 
     * Method: computeInternalVariables
     *
     * Purpose: 
     *   Update an old state with new values of internal variables given the old state and an 
     *   increment in volumetric plastic strain
     *
     * Inputs:
     *   state         - Old state
     *   delta_eps_p_v - negative in comression volumetruc plastic strain
     *
     * Outputs:
     *   state         - Modified state
     *
     * Returns:  true if success
     *           false if failure
     */
    //////////////////////////////////////////////////////////////////////////
    bool computeInternalVariables(ModelState_Arena& state,
                                  const double& delta_eps_p_v);

    //////////////////////////////////////////////////////////////////////////
    /** 
     * Method: computeDrainedHydrostaticStrengthAndDeriv
     *
     * Purpose: 
     *   Compute the drained hydrostatic compressive strength and its derivative
     *
     * Inputs:
     *   eps_bar_p_v - positive in comression volumetric plastic strain
     *   p3          - the value of p3 for a particle
     *
     * Outputs:
     *   Xbar_d      - drained hydrostatic compressive strength
     *   derivXbar_d - dXbar_d/dev_p
     */
    //////////////////////////////////////////////////////////////////////////
    void computeDrainedHydrostaticStrengthAndDeriv(const double& eps_bar_p_v,
                                                   const double& p3,
                                                   double& Xbar_d,
                                                   double& derivXbar_d) const;

    //////////////////////////////////////////////////////////////////////////
    /** 
     * Method: updateDamageParameters
     *
     * Purpose: 
     *   Update the damage parameters local to this model
     *
     * Inputs:
     *   D - the rate of defomation tensor
     *   delta_t - the time increment
     *   state_k_old - the state at the beginning of consistency bisection substep
     *   state_k_new - the state at the end of consistency bisection substep
     *
     * Outputs:
     *   state_k_new - state with updated damage parameters
     */
    //////////////////////////////////////////////////////////////////////////
    void updateDamageParameters(const Uintah::Matrix3& D,
                                const double& delta_t,
                                const ModelState_Arena& state_k_old,
                                ModelState_Arena& state_k_new) const;


  public: //Uintah MPM constitutive model specific functions

    // carry forward CM data for RigidMPM
    virtual void carryForward(const Uintah::PatchSubset* patches,
                              const Uintah::MPMMaterial* matl,
                              Uintah::DataWarehouse* old_dw,
                              Uintah::DataWarehouse* new_dw);


    // initialize  each particle's constitutive model data
    virtual void initializeCMData(const Uintah::Patch* patch,
                                  const Uintah::MPMMaterial* matl,
                                  Uintah::DataWarehouse* new_dw);


    ///////////////////////////////////////////////////////////////////////
    /*!
     * Actually initialize the stress and deformation gradient assuming linear
     * elastic behavior after computing the body force acceleration
     *
     * **WARNING** 1) Assumes zero shear stresses and that body forces are aligned
     *                with coordinate directions
     *             2) Needs the model to have a "initializeWithBodyForce" flag 
     *                set as true.  A more general implementation is not worth 
     *                the significant extra effort.
     */
    ///////////////////////////////////////////////////////////////////////
    /*
    void initializeStressAndDefGradFromBodyForce(const Uintah::Patch* patch,
                                                 const Uintah::MPMMaterial* matl,
                                                 Uintah::DataWarehouse* new_dw) const;
    */

    virtual void addInitialComputesAndRequires(Uintah::Task* task,
                                               const Uintah::MPMMaterial* matl,
                                               const Uintah::PatchSet* patches) const;

    virtual void addComputesAndRequires(Uintah::Task* task,
                                        const Uintah::MPMMaterial* matl,
                                        const Uintah::PatchSet* patches) const;

    virtual void addComputesAndRequires(Uintah::Task* task,
                                        const Uintah::MPMMaterial* matl,
                                        const Uintah::PatchSet* patches,
                                        const bool recursion,
                                        const bool dummy) const;

    virtual void addParticleState(std::vector<const Uintah::VarLabel*>& from,
                                  std::vector<const Uintah::VarLabel*>& to);

    virtual double computeRhoMicroCM(double pressure,
                                     const double p_ref,
                                     const Uintah::MPMMaterial* matl,
                                     double temperature,
                                     double rho_guess);

    virtual void computePressEOSCM(double rho_m, double& press_eos,
                                   double p_ref,
                                   double& dp_drho, double& ss_new,
                                   const Uintah::MPMMaterial* matl,
                                   double temperature);

    virtual double getCompressibility();

    /*! This is for adding/deleting particles when a particle is switched
      from one material to another */
    virtual void allocateCMDataAdd(Uintah::DataWarehouse* new_dw,
                                   Uintah::ParticleSubset* addset,
                                   Uintah::ParticleLabelVariableMap* newState,
                                   Uintah::ParticleSubset* delset,
                                   Uintah::DataWarehouse* old_dw);

  };
} // End namespace Uintah


#endif  // __ARENA_PARTIALLY_SATURATED__
